import numpy as np
import matlab

class BayesClassifier:

    def __init__(self):
        self.weights = None
        self.ntrees = 10
        self.nfeat = 10
        self.minVar = 300
        self.min_win = 8
        pass


    def config_weights(self, dens_prob):
        dens_prob = np.asarray(dens_prob)
        if dens_prob.shape[0] != self.ntrees:
            print("Error Classifier")
            return
        if dens_prob.shape[1] != 2 ** self.nfeat:
            print('Error Classifier')
            return
        self.weights = dens_prob
        pass

    def predict(self, descs, weights, roi):
        res = 0.0
        if len(descs) != 10:
            print('Error Desc Classifier')
        # if self.bbox_var_offset(iImg, iImg2, bbox) < self.minVar:
        # if np.var(roi) < self.minVar:
        #     conf = 0
        # else:
        conf = 0.0
        for i in range(0, self.ntrees):
            conf += weights[i][descs[i] - 1]
        # Проверить descs это массив из 10 чисел
        # находим среднее значение вероятности для всех 10 фернов
        # возвращаем результат - float 0...1.0
        #///10
        return (conf / 10)
