import time
import json
import torch

def calculate_time(time_taken):
    hrs, rem = divmod(int(time_taken), 3600)
    mins, secs = divmod(rem, 60)
    return hrs, mins, secs

class PerformanceTracker:

    def __init__(self, valid):

        self.valid = valid

        self.time_called = time.time()
        self.epoch_counter = 0
        self.forward_pass_counter = 0

        self.forward_pass_loss_tracker = self.reset_tracker()
        self.epoch_loss_tracker = self.reset_tracker()
        self.test_metric_tracker = []

        self.current_loss_tracker = self.reset_tracker()

    def reset_tracker(self):
        return {"train":None, "valid":None}
    
    def start_time_component(self):
        self.time_component = time.time()
    
    def end_time_component(self):
        return int(time.time() - self.time_component)
    
    def record_loss(self, loss, loss_type):
        if self.current_loss_tracker[loss_type] is None:
            self.current_loss_tracker[loss_type] = [loss]
        else:
            self.current_loss_tracker[loss_type].append(loss)
        self.forward_pass_counter += 1

    def record_test_performance(self, metric):
        self.test_metric_tracker.append(metric)

    def start_epoch(self):
        self.start_epoch_time = time.time()
        print(f"Epoch {self.epoch_counter + 1}:\n")
    
    def summary_stats(self):
        summ_stats = f"\t| Epoch {self.epoch_counter + 1} "
        summ_stats += f"| Pass {self.forward_pass_counter} "
        summ_stats += f"| Last training loss {self.current_loss_tracker['train'][-1]:.3f} "
        summ_stats += f"| Last validation loss {self.current_loss_tracker['valid'][-1]:.3f} " if self.valid else ""
        summ_stats += f"| Last test metric {self.test_metric_tracker[-1]:.3f} "

        epoch_runtime = time.time() - self.start_epoch_time
        total_runtime = time.time() - self.time_called

        hrs, mins, secs = calculate_time(epoch_runtime)
        summ_stats += f"| Epoch runtime {hrs}H:{mins}M:{secs}S"

        hrs, mins, secs = calculate_time(total_runtime)
        summ_stats += f"| Total runtime {hrs}H:{mins}M:{secs}S |"
        print(summ_stats)

    def end_epoch(self):
        
        for loss_type in self.current_loss_tracker:

            if self.forward_pass_loss_tracker[loss_type] is None:
                self.forward_pass_loss_tracker[loss_type] = self.current_loss_tracker[loss_type]
            else:
                self.forward_pass_loss_tracker[loss_type] += self.current_loss_tracker[loss_type]

            avg_epoch_loss = sum(self.current_loss_tracker[loss_type]) / len(self.current_loss_tracker[loss_type])
            if self.epoch_loss_tracker[loss_type] is None:
                self.epoch_loss_tracker[loss_type] = [avg_epoch_loss]
            else:
                self.epoch_loss_tracker[loss_type].append(avg_epoch_loss)

        self.current_loss_tracker = self.reset_tracker()
        self.forward_pass_counter = 0
        self.epoch_counter += 1
        self.epoch_time = time.time() - self.start_epoch_time
        self.total_runtime = time.time() - self.time_called

        summ_stats = f"\n\nEpoch {self.epoch_counter} completed...\n"
        summ_stats += f"Average training loss -- {self.epoch_loss_tracker['train'][-1]:.3f}\n"
        summ_stats += f"Average valid loss -- {self.epoch_loss_tracker['valid'][-1]:.3f}\n"
        summ_stats += f"Testing metric -- {self.test_metric_tracker[-1]:.3f}\n"
        
        hrs, mins, secs = calculate_time(self.epoch_time)
        summ_stats += f"Epoch runtime -- {hrs}H:{mins}M:{secs}S\n"

        hrs, mins, secs = calculate_time(self.total_runtime)
        summ_stats += f"Total runtime -- {hrs}H:{mins}M:{secs}S\n\n"
        print(summ_stats)
    
    def reset_pt(self):
        self.epoch_counter = 0
        self.forward_pass_counter = 0

        self.forward_pass_loss_tracker = self.reset_tracker()
        self.epoch_loss_tracker = self.reset_tracker()
        self.test_metric_tracker = []

        self.current_loss_tracker = self.reset_tracker()

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
    
    def save_model_results(self, path):

        data = {'forward pass losses': self.forward_pass_loss_tracker, 
                'average epoch losses': self.epoch_loss_tracker, 
                'test metrics': self.test_metric_tracker}
        
        with open(path, 'w') as f:
            json.dump(data, f)