class Layer_Training_Priority:

    def __init__(self, number_of_training_steps):
        self._training_steps = number_of_training_steps
        self._layers_record = [None for _ in range(self._training_steps)]

    def update_priority(self, td_error):
        # First entry in the record
        if self._layers_record[0] is None:
            self._layers_record[0] = td_error
            return self._training_steps
        else:
            iterations = 0
            for layer in range(self._training_steps):
                iterations +=1
                record = self._layers_record[layer]
                if record is None:
                    self._layers_record[layer] = td_error
                else:
                    if td_error > self._layers_record[layer]:
                        self._layers_record[layer] = td_error
                        break
            return self._training_steps - iterations
