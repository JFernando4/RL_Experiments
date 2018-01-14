class Layer_Training_Priority:

    def __init__(self, number_of_training_steps, record_size=10, number_of_percentiles=4, learning_rate=0.1):
        if number_of_percentiles > record_size:
            raise ValueError("The number of percentiles is greater than the record size.")
        self._training_steps = number_of_training_steps
            # record variables
        self._record_count = 0
        self._record_size = (record_size // 10) * 10            # a multiple of 10 for simplicity
        if self._record_size < 10: self._record_size = 10       # at least 10 entries in the record
            # percentiles variables
        self._record = [None for _ in range(self._record_size)]
        self._percentiles_full = False
        self._number_of_percentiles = number_of_percentiles
        self._percentiles = [0 for _ in range(self._number_of_percentiles)]
            # indexes with equal number of entries between them
        self._percentiles_indexes = [(i+1) * (self._record_size // (self._number_of_percentiles+1)) - 1
                                     for i in range(self._number_of_percentiles)]
        self._lr = learning_rate

    def update_priority(self, td_error):
        """ returns 0 to self._training_steps - 1 """
        self.add_to_record(td_error)
        if not self._percentiles_full:
            return 0 #self._training_steps - 1
        else:
            layer = (self._training_steps - 1) - self.get_closest_percentile_index(td_error)
            return layer

    def add_to_record(self, td_error):
        if self._record_count == self._record_size:
            self.reset_record()
        self._record_count += 1
        index = 0
        new_entry = td_error
        temp_entry = td_error
        while index < self._record_size:
            if self._record[index] is None:
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
        self._record = [None for _ in range(self._record_size)]

    def update_percentiles(self):
        if not self._percentiles_full:
            for index in range(self._number_of_percentiles):
                self._percentiles[index] = self._record[self._percentiles_indexes[index]]
            self._percentiles_full = True
        else:
            for index in range(self._number_of_percentiles):
                self._percentiles[index] += self._lr * (self._record[self._percentiles_indexes[index]]
                                                        - self._percentiles[index])

    def get_closest_percentile_index(self, td_error):
        smallest_difference = abs(self._percentiles[0] - td_error)
        smallest_index = 0
        for i in range(self._number_of_percentiles-1):
            abs_difference = abs(self._percentiles[i+1] - td_error)
            if abs_difference < smallest_difference:
                smallest_difference = abs_difference
                smallest_index = i + 1
        return smallest_index