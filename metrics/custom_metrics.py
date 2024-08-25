import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = tf.Variable(0, dtype = 'int32')
        self.fp = tf.Variable(0, dtype = 'int32')
        self.tn = tf.Variable(0, dtype = 'int32')
        self.fn = tf.Variable(0, dtype = 'int32')

    def update_state(self, y_true, y_pred, sample_weight=None):
        conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)
        self.tn.assign_add(conf_matrix[0][0])
        self.tp.assign_add(conf_matrix[1][1])
        self.fp.assign_add(conf_matrix[0][1])
        self.fn.assign_add(conf_matrix[1][0])

    def result(self):
        if (self.tp + self.fp == 0):
            precision = 1.0
        else:
            precision = self.tp / (self.tp + self.fp)
        if (self.tp + self.fn == 0):
            recall = 1.0
        else:
            recall = self.tp / (self.tp + self.fn)
        
        if (self.tn + self.fp == 0):
            specificity=1.0
        else:
            specificity= self.fp/ (self.tn + self.fp)
            
        
        f1_score = 2*((precision*recall)/(precision + recall))
  
        return [f1_score, precision, recall, specificity]

    def reset_states(self):
        self.tp.assign(0)
        self.tn.assign(0) 
        self.fp.assign(0)
        self.fn.assign(0)