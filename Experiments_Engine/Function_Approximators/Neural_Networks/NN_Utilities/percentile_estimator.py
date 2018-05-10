import numpy as np


class Percentile_Estimator:

    def __init__(self, number_of_percentiles=4, learning_rate=0.01):
        self._number_of_percentiles = number_of_percentiles
        self._lr = learning_rate
        self._percentiles = np.zeros(self._number_of_percentiles)
        self._record = np.zeros(self._number_of_percentiles)
        self._record_count = 0
        self._record_empty = True

    def add_to_record(self, td_error):
        if self._record_count == self._number_of_percentiles:
            self.reset_record()
        self._record_count += 1
        index = 0
        new_entry = td_error
        temp_entry = td_error
        while index < self._number_of_percentiles:
            if index == (self._record_count -1):
                self._record[index] = new_entry
                break
            else:
                if new_entry > self._record[index]:
                    temp_entry = self._record[index]
                    self._record[index] = new_entry
                    new_entry = temp_entry
                index += 1

    def reset_record(self):
        self.update_percentiles()
        self._record_count = 0
        self._record = np.zeros(self._number_of_percentiles)

    def update_percentiles(self):
        if not self._record_empty:
            self._percentiles += self._lr * (self._record - self._percentiles)
        else:
            self._percentiles += self._record
            self._record_empty = False
        # print("The percentiles are:", self._percentiles)

    def get_percentile(self, index):
        if self._number_of_percentiles == 0:
            return -np.inf
        if index > self._number_of_percentiles-1:
            raise ValueError("The index needs to be a number in [0, number_of_percentiles - 1]")
        return self._percentiles[index]
