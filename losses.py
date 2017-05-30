"""
  Computes the Batch Cross-Entropy loss, as appears in:
  "Fast Single-Class Classification and the Principle of Logit Separation",
  Gil Keren, Sivan Sabato and Bjoern Schuller.
  (https://arxiv.org/abs/1705.10246)


  Args:
    logits: The logits for the batch, of shape `[batch_size, num_classes]`
    labels: The labels for the batch, of shape `[batch_size]` containing class indices, 
            or `[batch_size, num_classes]` for one hot encoding.

  Returns:
    The batch cross-entropy loss
"""
def batch_ce(logits, labels):
  epsilon = 1e-6
  batch_size = tf.shape(logits)[0]
  n_classes = logits.shape[1].value
  
  # Convert labels to one hot encoding
  if len(labels.shape) == 1:
    labels = tf.one_hot(labels, n_classes)
  
  # The model output distribution Q
  Q = tf.nn.softmax(tf.reshape(logits, [-1]))
  
  # Mask for true logits
  true_mask = tf.cast(labels, 
                      tf.bool)
  
  # The target distribution P
  P = tf.where(true_mask, 
              tf.ones_like(logits, tf.float32) / tf.to_float(batch_size), 
              tf.zeros_like(logits, tf.float32))
  P = tf.reshape(P, [-1])
  
  # The batch cross-entropy loss is the KL divergence between P and Q
  # Epsilon is added for numerical stability
  KL = tf.reduce_sum(tf.log(P / (Q + epsilon) + epsilon) * P)
  return KL


"""
  Computes the Batch Cross-Entropy loss, as appears in:
  "Fast Single-Class Classification and the Principle of Logit Separation",
  Gil Keren, Sivan Sabato and Bjoern Schuller.
  (https://arxiv.org/abs/1705.10246)

  Args:
    logits: The logits for the batch, of shape `[batch_size, num_classes]`
    labels: The labels for the batch, of shape `[batch_size]` containing class indices, 
            or `[batch_size, num_classes]` for one hot encoding.

  Returns:
    The batch max-margin loss
"""
def batch_max_margin_final(logits, labels):
  batch_size = tf.shape(logits)[0]
  n_classes = logits.shape[1].value
  
  # Convert labels to one hot encoding
  if len(labels.shape) == 1:
    labels = tf.one_hot(labels, n_classes)
  
  # Mask for true logits
  true_mask = tf.cast(labels, 
                      tf.bool)
  
  # True and false logits 
  true_logits = tf.boolean_mask(logits, true_mask)
  false_logits = tf.boolean_mask(logits, tf.logical_not(true_mask))
  false_logits = tf.reshape(false_logits, [batch_size, n_classes-1])
  
  # Max false logit per example
  false_logits_example_max = tf.reduce_max(false_logits, axis=1)
  
  # Max of false logits and Min of correct logits
  true_logits_min = tf.reduce_min(true_logits)
  false_logits_max = tf.reduce_max(false_logits)
  
  # True to false logits difference, per example and per batch
  example_diff = true_logits - false_logits_example_max
  batch_diff = true_logits_min - false_logits_max
  
  # The batch max-margin loss
  return tf.reduce_mean(tf.maximum(0.0, FLAGS.gamma - example_diff)) + \
         tf.maximum(0.0, FLAGS.gamma - batch_diff) * (1.0 / tf.to_float(batch_size))

