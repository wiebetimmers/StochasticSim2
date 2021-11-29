import random
import os
from tqdm import tqdm
import statistics as st
from simpy import *
import simpy
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
fig = plt.figure(figsize=(6,4), dpi=300)

# https://medium.com/swlh/simulating-a-parallel-queueing-system-with-simpy-6b7fcb6b1ca1
# https://simpy.readthedocs.io/en/latest/examples/bank_renege.html



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


def source(env, number, counters, serv_dist):
    """Source generates customers randomly"""
    for i in range(number):
        c = customer(env, counters, serv_dist, name="customer_%s"%i)
        env.process(c)
        t = random.expovariate(ARR_RATE/len(counters))  # divide by len(counters) to obtain n-fold lower arr. rate
        yield env.timeout(t)

def no_in_sys(r):
    return max([0, len(r.put_queue) + len(r.users)])

def customer(env, counters, serv_dist, name):
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
        if serv_dist == 'M':
            tib = random.expovariate(SERV_RATE)  # divide by cap as serv rate decreases if cap rises
        elif serv_dist == 'D':
            tib = INTERVAL_SERVICE
        elif serv_dist == 'H':
            determiner = random.uniform(0,1)
            if determiner <= 0.75:
                tib = random.expovariate(1.0 / 1.0) # avg service time of 1.0
            else:
                tib = random.expovariate(1.0 / 5.0) # avg service time of 5.0
        yield env.timeout(tib)
        end = env.now
        #print('%s got finished at %s' % (name, end))
        service_time = tib
        sojourn_time = tib + wait
        dists["M%s%s" % (serv_dist, len(counters))]['wait_times'].append(wait)
        dists["M%s%s" % (serv_dist, len(counters))]['serv_times'].append(service_time)
        dists["M%s%s" % (serv_dist, len(counters))]['soj_times'].append(sojourn_time)
        return
        #print('%7.4f %s: Sojourn Time' % (end-arrive, name))
        #print('%7.4f %s: Finished' % (env.now, name))


def run_simulation(capacities, service_dis):
    es_list = []
    wait_list = []
    # Run the analytical solution once per cap
    for cap in tqdm(capacities):
        p = system_load(cap)
        el = (p) / (1 - p)
        es = (1 / (SERV_RATE * cap)) / (1 - p)
        wait_time = es - (1 / (SERV_RATE * cap))
        print('System load: ', p)
        print('Mean number customers in system: ', el)
        print('Mean sojourn time: ', es)
        print('Mean wait time: ', wait_time)
        print('Mean service time: ', INTERVAL_SERVICE)
        print('-------------------')
        es_list.append(es)
        wait_list.append(wait_time)

        # Run the simulations for every cap and service dist
        for sd in service_dis:
            random.seed(RANDOM_SEED)
            env = simpy.Environment()
            counters = []
            for i in range(cap):
                counters.append(Resource(env))
            env.process(source(env, NEW_CUSTOMERS, counters, serv_dist=sd))
            env.run()
    return es_list, wait_list

def plot_functions(results, capacities, metrics, service_dis):
    print('Plotting ....')
    for met in tqdm(metrics):
        for cap in capacities:
            for sd in service_dis:
                plt.clf()
                ax = sns.displot(results['M%s%s'%(sd, cap)][met], label='n=%s'%cap)
                ax.set(xlabel='%s'%met, ylabel=('Frequency'))
                plt.tight_layout()
                if not os.path.exists('displots/M%s%s'%(sd, cap)):
                    os.makedirs('displots/M%s%s'%(sd, cap))
                plt.savefig('displots/M%s%s/%s_n_%s.jpg' %(sd, cap, met, NEW_CUSTOMERS))
                plt.close()
    return

def get_stats(results, capacities, metrics, service_dis):
    for cap in capacities:
        for met in metrics:
            for sd in service_dis:
                cap_met = results['M%s%s'%(sd, cap)][met]
                results['M%s%s'%(sd, cap)]['mean_%s' %met] = st.mean(cap_met)
                results['M%s%s'%(sd, cap)]['max_%s' %met] = max(cap_met)
                print('mean M%s%s, met %s: %s'%(sd, cap, met, st.mean(cap_met)))
    file_to_write = open("stats.pickle", "wb")
    pkl.dump(results, file_to_write)
    file_to_write.close()
    return

def make_table(file):
    infile = open(file, 'rb')
    stats = pkl.load(infile)
    df = pd.DataFrame.from_dict(stats)
    df = df.drop(['wait_times', 'serv_times', 'soj_times'])
    df2 = df.transpose()
    print(df2)
    df2.to_excel("output.xlsx")
    return

# Define domains to simulate
capacities = [1, 2, 4]
service_dis = ['M', 'D', 'H']
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
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    },
    'MD1' : {
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    },
    'MD2' : {
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    },
    'MD4' : {
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    },
    'MH1': {
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    },
    'MH2': {
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    },
    'MH4': {
        'wait_times' : [],
        'serv_times' : [],
        'soj_times' : []
    }}


# The script subsections, comment out for partial run.
es_list, wait_list = run_simulation(capacities, service_dis)
plot_functions(dists, capacities, metrics, service_dis)
get_stats(dists, capacities, metrics, service_dis)
make_table("stats.pickle")

plt.clf()
plt.plot(capacities, es_list, label='E(S)')
plt.plot(capacities, wait_list, label='E(W)')
plt.legend()
plt.xlabel('Capacity')
plt.ylabel('Time')
plt.savefig('ew_es_plot.jpg')
