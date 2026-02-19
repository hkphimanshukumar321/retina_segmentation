
import tensorflow as tf

class DiceScore(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name="dice_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.dice_sum = self.add_weight(name="dice_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Handle deep supervision (dict output) if metrics applied blindly
        # But usually Keras applies per-output. We assume y_pred is the Tensor for this head.
        
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        # Threshold
        y_pred_bin = tf.cast(y_pred > 0.5, y_pred.dtype)
        
        # Compute Dice per sample, per class
        # shape: (B, H, W, C)
        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2])
        union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred_bin, axis=[1, 2])
        
        # Dice = 2*I / (U + eps)
        # Avoid div by zero. If union is 0, dice is technically undefined or 1.0 if both empty.
        # Here we treat empty-empty as 1.0 (perfect prediction of background)
        epsilon = 1e-5
        dice_score = (2.0 * intersection + epsilon) / (union + epsilon)
        
        # Mean over classes and batch
        batch_dice = tf.reduce_mean(dice_score) # Scalar
        
        self.dice_sum.assign_add(batch_dice)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.dice_sum, self.count)

    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)

class IoUScore(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name="iou_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.iou_sum = self.add_weight(name="iou_sum", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred_bin = tf.cast(y_pred > 0.5, y_pred.dtype)
        
        intersection = tf.reduce_sum(y_true * y_pred_bin, axis=[1, 2])
        union = tf.reduce_sum(y_true + y_pred_bin, axis=[1, 2]) - intersection
        # or simplified: sum(A)+sum(B)-sum(AB)
        # binary: sum(A | B)
        
        epsilon = 1e-5
        iou_score = (intersection + epsilon) / (union + epsilon)
        
        batch_iou = tf.reduce_mean(iou_score)
        
        self.iou_sum.assign_add(batch_iou)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.iou_sum, self.count)

    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)
