"""
Bank renege example

Covers:

- Resources: Resource
- Condition events

Scenario:
  A counter with a random service time and customers who renege. Based on the
  program bank08.py from TheBank tutorial of SimPy 2. (KGM)

"""
import random
from tqdm import tqdm
import statistics as st
from simpy import *
import simpy
import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,4), dpi=300)


RANDOM_SEED = 12345
NEW_CUSTOMERS = 100000 # Total number of customers
INTERVAL_CUSTOMERS = 10.0  # Generate new customers roughly every x seconds
INTERVAL_SERVICE = 9.5 # Service takes roughly 10 seconds of time

ARR_RATE = 1/INTERVAL_CUSTOMERS  #lambda
SERV_RATE = 1/INTERVAL_SERVICE   #mu

def system_load(n):
    # We assume the same load characteristics for every experiment:
    # This means the arrival rate is set to be n-fold lower.
    p = (ARR_RATE/n) / (n*SERV_RATE)
    return p


def source(env, number, counters):
    """Source generates customers randomly"""
    for i in tqdm(range(number)):
        c = customer(env, counters, name="customer_%s"%i)
        env.process(c)
        t = random.expovariate(ARR_RATE/len(counters))  # divide by len(counters) to obtain n-fold lower arr. rate
        yield env.timeout(t)

def no_in_sys(r):
    return max([0, len(r.put_queue) + len(r.users)])

def customer(env, counters, name):
    """Customer arrives, waits, is served and leaves."""
    arrive = env.now
    #print('%s arrives at %s' %(name, arrive))
    queue_length = [no_in_sys(counters[i]) for i in range(len(counters))]
    #print('queue length: ', queue_length)
    choice = 0
    for i in range(len(queue_length)):
        if queue_length[i] == 0 or queue_length[i] == min(queue_length):
            choice = i  # the chosen queue number
            break
    # wait till counter becomes available:
    with counters[choice].request() as req:
        yield req
        #print('Queue size: ', len(counters[choice].queue))
        wait = env.now - arrive
        #print('%s has waited %s seconds' % (name, wait ))
        tib = random.expovariate(SERV_RATE/len(counters))  # divide by cap as serv rate decreases if cap rises
        yield env.timeout(tib)
        end = env.now
        #print('%s got finished at %s' % (name, end))
        service_time = tib
        sojourn_time = tib + wait
        dists["MM%s" % len(counters)]['wait_times'].append(wait)
        dists["MM%s" % len(counters)]['serv_times'].append(service_time)
        dists["MM%s" % len(counters)]['soj_times'].append(sojourn_time)
        return
        #print('%7.4f %s: Sojourn Time' % (end-arrive, name))
        #print('%7.4f %s: Finished' % (env.now, name))


# Start processes and run
capacities = [1, 2, 4]
es_list = []
wait_list = []
metrics = ['wait_times', 'serv_times', 'soj_times']
dists = {
    'MM1' : {
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    },
    'MM2' : {
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    },
    'MM4' : {
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    },
    'MM1_short_job_first' :{
    },
    'MD1' : {
    },
    'MD2' : {
    },
    'MD4' : {
    },
    'M_hyperexp_1': {
    },
    'M_hyperexp_2': {
    },
    'M_hyperexp_4': {
    }}

for cap in capacities:
    p = system_load(cap)
    el = (p)/(1-p)
    es = (1/(SERV_RATE*cap))/(1-p)
    wait_time = es - (1/(SERV_RATE*cap))
    print('System load: ', p)
    print('Mean number customers in system: ', el)
    print('Mean sojourn time: ', es)
    print('Mean wait time: ', wait_time)
    print('Mean service time: ', es-wait_time)
    print('-------------------')
    es_list.append(es)
    wait_list.append(wait_time)
    # Setup and start the simulation
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    # Perform experiments
    counters = []
    for i in range(cap):
        counters.append(Resource(env))
    env.process(source(env, NEW_CUSTOMERS, counters))
    env.run()

def plot_functions(results, capacities, metrics):
    for met in metrics:
        plt.clf()
        for cap in capacities[::-1]:
            plt.hist(results['MM%s'%cap][met], label='n=%s'%cap)
            plt.xlabel('%s'%met)
            plt.ylabel('Frequency')
            plt.legend()
        plt.savefig('displots/%s_n_%s.jpg' %(met, NEW_CUSTOMERS))
    return

def get_stats(results, capacities, metrics):
    for cap in capacities:
        for met in metrics:
            cap_met = results['MM%s'%cap][met]
            results['MM%s' % cap]['mean_%s' %met] = st.mean(cap_met)
            results['MM%s' % cap]['max_%s' %met] = max(cap_met)
            print('mean MM%s, met %s: %s'%(cap, met, st.mean(cap_met)))
    return

plot_functions(dists, capacities, metrics)
get_stats(dists, capacities, metrics)


plt.clf()
plt.plot(capacities, es_list, label='E(S)')
plt.plot(capacities, wait_list, label='E(W)')
plt.legend()
plt.xlabel('Capacity')
plt.ylabel('Time')
plt.savefig('expdecay_sojourningtime.jpg')
