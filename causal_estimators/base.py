from abc import ABC, abstractmethod


class BaseEstimator(ABC):

    @abstractmethod
    def fit(self, w, t, y):
        pass

    @abstractmethod
    def estimate_ate(self, t1=1, t0=0, w=None):
        pass

    @abstractmethod
    def ate_conf_int(self, percentile=.95) -> tuple:
        pass


class BaseIteEstimator(BaseEstimator):

    @abstractmethod
    def fit(self, w, t, y):
        pass

    @abstractmethod
    def predict_outcome(self, t, w):
        pass

    def estimate_ate(self, t1=1, t0=0, w=None):
        return self.estimate_ite(t1=t1, t0=t0, w=w).mean()

    @abstractmethod
    def ate_conf_int(self, percentile=.95):
        pass

    def estimate_ite(self, t1=1, t0=0, w=None):
        return self.predict_outcome(t=t1, w=w) - self.predict_outcome(t=t0, w=w)

    # TODO: ITE confidence interval