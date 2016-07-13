import os
import multiprocessing
from tqdm import tqdm

def saveOrDisplay(save, file_name):
    """outputs False or a filename to save a plot"""
    if save:
        f_name = os.path.basename(file_name)
        save_name = os.path.splitext(f_name)[0] + '.png'
    else:
        save_name = False
    return save_name
#
def target_func(f, q_in, q_out):
    """Used in parmap - I don't know / care what this does, as it works
    see: http://stackoverflow.com/q/3288595/4013571 for more info"""
    while not q_in.empty():
        i, x = q_in.get()
        q_out.put((i, f(x)))

#
def prll_map(function_to_apply, items, cpus=None, verbose=False):
    """multicore support for instanced class funcs. default
    args uses all available cores
    
    adapted from: http://stackoverflow.com/a/37499872/4013571
    with the support for progress bars
    
    Required Inputs
        function_to_apply   :: func :: this function can within an instance of a class
        items   :: an input to the function_to_apply - should allow multiple item inputs
    
    Optional Inputs
        cpus    :: int   :: number of processors to use: default is ALL
        verbose :: bool  :: progress bar
    """
    if cpus is None: cpus = min(multiprocessing.cpu_count(), 32)
    print '> optimised for {} processors'.format(cpus)
    
    # Create queues
    q_in  = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    
    # Process list #
    new_proc  = lambda t,a: multiprocessing.Process(target=t, args=a)
    processes = [new_proc(target_func, (function_to_apply, q_in, q_out)) for x in range(cpus)]
    
    # Queue processes
    sent = [q_in.put((i, x)) for i, x in enumerate(items)]
    
    for proc in processes: # Start them all
        proc.daemon = True
        proc.start()
    
    # Result from out queue
    if verbose:
        result = [q_out.get() for _ in tqdm(sent)]
    else: # tqdm prints display
        result = [q_out.get() for _ in sent]
    for proc in processes: proc.join() # Wait for them to finish
    
    return [x for i, x in sorted(result)] # Return results sorted