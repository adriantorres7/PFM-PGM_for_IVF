import numpy as np

class EM:
    def __init__(self, model, max_its=100, tolerance=1e-8):
        self.model = model
        self.max_iterations = max_its
        self.tol = tolerance

        self.estimation_record = []
        self.model_record = []

    # Run EM algorithm
    def run(self):
        self.initialize()
        while not self.test_convergence():
            #print("   EM iteration num.", self.it_em)
            self.expectation()
            self.maximization()
        print('Iterations: ', self.it_em)

        #print("   Finish!")

    # Initialize
    def initialize(self):
        self.it_em = 0
        self.estimation_record.append(self.model.initialize())
        self.model_record.append(self.model.fit())  # fit_class_embryos, fit_class_cycles, fit_thetas

    # Expectation EM Step
    def expectation(self):
        self.estimation_record.append(self.model.estimations())

    # Maximization EM Step
    def maximization(self):
        self.model_record.append(self.model.fit())  # fit_class_embryos, fit_class_cycles, fit_thetas


    # Evaluates convergence
    def test_convergence(self):
        convergence = False
        self.it_em += 1
        if self.it_em == 1:
            convergence = False
        elif self.compare_two_last_estimations() or self.it_em == self.max_iterations:
            convergence = True
        return convergence

    def compare_two_last_estimations(self):
        old_est = self.estimation_record[self.it_em-1]
        act_est = self.estimation_record[self.it_em-2]

        for old_m, act_m in zip(old_est, act_est):
            if not np.allclose(old_m, act_m, atol=self.tol):
                return False

        return True

