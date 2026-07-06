import torch
import torch.nn as nn
import torch.optim as optim


class ScopeClassifier:

    def __init__(self, model, hyperparameters):

        criterion_name = hyperparameters.get("criterion", "cross_entropy")
        if criterion_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_name == "nll":
            self.criterion = nn.NLLLoss()
        else:
            raise ValueError("Unknown criterion: " + str(criterion_name))

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters.get("weight_decay", 0.0),
        )

    def compute_loss(self, y_hat, y):

        return self.criterion(y_hat, y)


class ScopeBLFormer:

    def __init__(self, model, hyperparameters):

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparameters["learning_rate"],
            weight_decay=hyperparameters.get("weight_decay", 0.0),
        )

        self.potted_head = hyperparameters["potted_head"]
        self.clear_weight = hyperparameters.get("clear_weight", 1.0)
        self.win_weight = hyperparameters.get("win_weight", 1.0)
        self.potted_weight = hyperparameters.get("potted_weight", 1.0)
        self.potted_ordinal_weight = hyperparameters.get("potted_ordinal_weight", 0.25)
        self.use_joint_head = hyperparameters.get("use_joint_head", False)
        self.joint_marginal_weight = hyperparameters.get("joint_marginal_weight", 0.0)
        self.potted_class_weights = hyperparameters.get("potted_class_weights", None)

    def coral_targets(self, y, num_thresholds):

        thresholds = torch.arange(num_thresholds, device=y.device)
        return (y[:, None] > thresholds[None, :]).float()

    def compute_loss(self, outputs, batch):

        clear_target = batch["clear"].float()
        win_target = batch["win"].float()
        potted_target = self.coral_targets(
            batch["potted_after_break"],
            outputs["potted_logits"].shape[1],
        )

        clear_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["clear_logit"],
            clear_target,
        )
        win_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["win_logit"],
            win_target,
        )
        ordinal_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["potted_logits"],
            potted_target,
            reduction="none",
        )
        ordinal_loss = ordinal_loss.sum(dim=1).mean()

        class_weights = self.potted_class_weights
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        if class_weights is not None:
            class_weights = class_weights.to(outputs["potted_class_logits"].device)

        class_loss = nn.functional.cross_entropy(
            outputs["potted_class_logits"],
            batch["potted_after_break"],
            weight=class_weights,
        )

        if self.potted_head == "ordinal":
            potted_loss = ordinal_loss
        elif self.potted_head == "class":
            potted_loss = class_loss
        elif self.potted_head == "hybrid":
            potted_loss = class_loss + self.potted_ordinal_weight * ordinal_loss
        else:
            raise ValueError("Unknown potted head: " + str(self.potted_head))

        total_loss = (
            self.clear_weight * clear_loss
            + self.win_weight * win_loss
            + self.potted_weight * potted_loss
        )
        joint_loss = total_loss.new_tensor(0.0)

        if self.use_joint_head:
            joint_target = (
                batch["clear"].long() * 20
                + batch["win"].long() * 10
                + batch["potted_after_break"].long()
            )
            joint_loss = nn.functional.cross_entropy(outputs["joint_logits"], joint_target)
            marginal_loss = (clear_loss + win_loss + class_loss) / 3.0
            total_loss = joint_loss + self.joint_marginal_weight * marginal_loss

        return {
            "loss": total_loss,
            "clear_loss": clear_loss.detach(),
            "win_loss": win_loss.detach(),
            "potted_loss": potted_loss.detach(),
            "potted_ordinal_loss": ordinal_loss.detach(),
            "potted_class_loss": class_loss.detach(),
            "joint_loss": joint_loss.detach(),
        }

    def predict(self, outputs):

        clear_pred = (outputs["clear_logit"] > 0.0).long()
        win_pred = (outputs["win_logit"] > 0.0).long()

        if self.potted_head == "ordinal" and not self.use_joint_head:
            potted_pred = (outputs["potted_logits"] > 0.0).sum(dim=1).long()
        else:
            potted_pred = torch.argmax(outputs["potted_class_logits"], dim=1)

        return {
            "clear": clear_pred,
            "win": win_pred,
            "potted_after_break": potted_pred,
        }
