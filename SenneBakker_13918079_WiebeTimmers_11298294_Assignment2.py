import random
import os
import simpy
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import statistics as st
from tqdm import tqdm
from simpy import *
from scipy import stats

fig = plt.figure(figsize=(6,4), dpi=300)

# Sources for inspiration
# https://simpy.readthedocs.io/en/latest/examples/bank_renege.html
# https://www.programiz.com/dsa/insertion-sort


def system_load(n):
    # We assume the same load characteristics for every experiment:
    # This means the arrival rate is set to be n-fold lower.
    p = (ARR_RATE*n) / (n*SERV_RATE)
    return p

def insertionsort(array):
    for step in range(1, len(array)):
        key = array[step]
        j = step - 1
        # Compare key with each element on the left of it until an element smaller than it is found
        # For descending order, change key<array[j] to key>array[j].
        while j >= 0 and key < array[j]:
            array[j + 1] = array[j]
            j = j - 1

        # Place key at after the element just smaller than it.
        array[j + 1] = key
    return array

def source(env, number, counters, serv_dist):
    sjf_jobs = []
    """Source generates customers randomly"""
    for i in range(number):
        if serv_dist == 'MM1_sjf':
            sjf_jobs.append(random.expovariate(SERV_RATE))
            sjf_jobs = insertionsort(sjf_jobs)
        c = customer(env, counters, serv_dist, sjf_jobs, name="customer_%s"%i)
        env.process(c)
        t = random.expovariate(ARR_RATE*len(counters))  # multiply by n , to keep system load constant
        yield env.timeout(t)

def no_in_sys(r):
    return max([0, len(r.put_queue) + len(r.users)])

def customer(env, counters, serv_dist, sjf_jobs, name):
    """Customer arrives, waits, is served and leaves."""
    arrive = env.now
    #print('%s arrives at %s' %(name, arrive))
    queue_length = [no_in_sys(counters[i]) for i in range(len(counters))]
    #print('queue length: ', queue_length)
    choice = 0

    # Pick queue with shortest length:
    for i in range(len(queue_length)):
        if queue_length[i] == 0 or queue_length[i] == min(queue_length):
            choice = i
            break

    # wait till counter becomes available:
    with counters[choice].request() as req:
        yield req
        #print('queue:', counters[choice].queue)
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
                tib = random.expovariate(1.0 / 5.0) # avg service time of 5.0
            else:
                tib = random.expovariate(1.0 / 15.0) # avg service time of 15.0 - jobs that just take  longer
        elif serv_dist == 'MM1_sjf':
            tib = sjf_jobs[0]
            sjf_jobs.pop(0)
        yield env.timeout(tib)
        if serv_dist == 'MM1_sjf':
            dists["MM1_sjf"]['wait_times'].append(wait)
        else:
            dists["M%s%s" % (serv_dist, len(counters))]['wait_times'].append(wait)
        return

def run_analytical_simulation(capacities):
    es_list = []
    wait_list = []
    # Run the analytical solution once per cap
    for cap in capacities:
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
    return es_list, wait_list

def run_simulation(capacities, service_dis):
    # Run the analytical solution once per cap
    for sd in tqdm(service_dis):
        for cap in capacities:
            print('\nsimulating M%s%s'%(sd, cap))
            print('--------')
            # Run the simulations 100 times for every cap and service dist
            for i in range(SAMPLE_SIZE_SIM):
                env = simpy.Environment()
                counters = []
                for i in range(cap):
                    counters.append(Resource(env))
                env.process(source(env, NEW_CUSTOMERS, counters, serv_dist=sd))
                env.run()
                if sd == 'MM1_sjf':
                    ew_mean = st.mean(dists['MM1_sjf']['wait_times'])
                    dists['MM1_sjf']['mean_w_times'].append(ew_mean)
                    dists['MM1_sjf']['wait_times'] = []
                else:
                    ew_mean = st.mean(dists["M%s%s" % (sd, cap)]['wait_times'])
                    dists["M%s%s" % (sd, cap)]['mean_w_times'].append(ew_mean)
                    #Empty wait times again for next run
                    dists["M%s%s" % (sd, cap)]['wait_times'] = []
            if sd == 'MM1_sjf':
                break

    file_to_write = open("data.pickle", "wb")
    pkl.dump(dists, file_to_write)
    file_to_write.close()
    return

def plot_boxplots(bpdict):
    dictfilt = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
    cap1 = dictfilt(bpdict, ['MM1','MM1_sjf','MD1','MH1'])
    cap2 = dictfilt(bpdict, ['MM2', 'MD2', 'MH2'])
    cap4 = dictfilt(bpdict, ['MM4', 'MD4', 'MH4'])

    names = ['cap1', 'cap2', 'cap4']
    count = 0
    for i in [cap1, cap2, cap4]:
        fig, ax = plt.subplots()
        plt.ylabel('Average Waiting Time')
        plt.xlabel('Method')
        #for c in cap1:
        ax.boxplot(i.values(), showfliers=False, patch_artist=True)
        ax.set_xticklabels(i.keys())
        plt.savefig('boxplot_%s.jpg'%str(names[count]))
        plt.close()
        count +=1
    return

def plot_functions(results, capacities, metrics, service_dis):
    boxplotdict = {}
    print('Plotting ....')
    for met in metrics:
        for sd in service_dis:
            for cap in capacities:
                plt.clf()
                if sd == "MM1_sjf":
                    ax = sns.displot(results["MM1_sjf"][met], label='n=%s' % cap)
                    boxplotdict["MM1_sjf"] = results["MM1_sjf"][met]
                else:
                    ax = sns.displot(results['M%s%s'%(sd, cap)][met], label='n=%s'%cap)
                    boxplotdict['M%s%s'%(sd, cap)] = results['M%s%s'%(sd, cap)][met]
                ax.set(xlabel='%s'%met, ylabel=('Frequency'))
                plt.tight_layout()
                plt.savefig('displots/M%s%s_%s_n_%s.jpg' %(sd, cap, met, NEW_CUSTOMERS))
                plt.close()
                if sd == "MM1_sjf":
                    break

    plot_boxplots(boxplotdict)
    return

def plot_analytic_result(es_list, wait_list, capacities):
    plt.plot(capacities, es_list, label='E(S)')
    plt.plot(capacities, wait_list, label='E(W)')
    plt.xlabel('Capacity')
    plt.ylabel('Time')
    plt.legend()
    plt.savefig('analytic.jpg')
    plt.close()
    return

def get_stats(results, capacities, metrics, service_dis):
    for sd in service_dis:
        for met in metrics:
            for cap in capacities:
                if sd == 'MM1_sjf':
                    cap_met = results['MM1_sjf'][met]
                    results['MM1_sjf']['mean_%s' % met] = st.mean(cap_met)
                    results['MM1_sjf']['stdev_%s' % met] = st.stdev(cap_met)
                    results['MM1_sjf']['max_%s' % met] = max(cap_met)
                    results['MM1_sjf']['min_%s' % met] = min(cap_met)
                    print('mean MM1_sjf, met %s: %s' % (met, st.mean(cap_met)))
                    break
                else:
                    cap_met = results['M%s%s'%(sd, cap)][met]
                    results['M%s%s'%(sd, cap)]['mean_%s' %met] = st.mean(cap_met)
                    results['M%s%s' % (sd, cap)]['stdev_%s' % met] = st.stdev(cap_met)
                    results['M%s%s'%(sd, cap)]['max_%s' %met] = max(cap_met)
                    results['M%s%s' % (sd, cap)]['min_%s' % met] = min(cap_met)
                    print('mean M%s%s, met %s: %s'%(sd, cap, met, st.mean(cap_met)))
    file_to_write = open("stats.pickle", "wb")
    pkl.dump(results, file_to_write)
    file_to_write.close()
    return

def do_stat_test(file):
    methods = pd.Series(['MM1', 'MM2', 'MM4', 'MM1_sjf', 'MD1', 'MD2', 'MD4', 'MH1', 'MH2', 'MH4'])
    infile = open(file, 'rb')
    data = pkl.load(infile)
    # fill in p.value for every combination, we do a Welsh t-test and test if the samples are different
    # We set our p crit value to 0.01
    # H0: method x mean = method y mean
    # H1: method x mean != method y mean
    df = pd.DataFrame(methods.apply(lambda x: methods.apply(lambda y: (stats.ttest_ind(data[x]['mean_w_times'],
                                                                                      data[y]['mean_w_times'],
                                                                                      equal_var=False)[1]))))
    df.index = methods
    df.columns = methods
    df.to_excel('stat_test_output.xlsx')
    return

def make_table(file):
    infile = open(file, 'rb')
    data = pkl.load(infile)
    df = pd.DataFrame.from_dict(data)
    df = df.drop(['wait_times', 'mean_w_times'])
    df2 = df.transpose()
    print(df2)
    df2.to_excel("output.xlsx")
    return

# Define domains to simulate
capacities = [1, 2, 4]
service_dis = ['MM1_sjf', 'M', 'D', 'H']
metrics = ['mean_w_times']
dists = {
    'MM1' : {
        'wait_times' : [],
        'mean_w_times': []
    },
    'MM2' : {
        'wait_times' : [],
        'mean_w_times': []
    },
    'MM4' : {
        'wait_times' : [],
        'mean_w_times': []
    },
    'MM1_sjf' :{
        'wait_times' : [],
        'mean_w_times': []
    },
    'MD1' : {
        'wait_times' : [],
        'mean_w_times': []
    },
    'MD2' : {
        'wait_times' : [],
        'mean_w_times': []
    },
    'MD4' : {
        'wait_times' : [],
        'mean_w_times': []
    },
    'MH1': {
        'wait_times' : [],
        'mean_w_times': []
    },
    'MH2': {
        'wait_times' : [],
        'mean_w_times': []
    },
    'MH4': {
        'wait_times' : [],
        'mean_w_times': []
    }}

# Parameters
SAMPLE_SIZE_SIM = 1000   # no simulation to obtain a distribution of E(w)
RANDOM_SEED = 12345
NEW_CUSTOMERS = 10000 # Total number of customers
INTERVAL_CUSTOMERS = 10.0  # Generate new customers roughly every x seconds
INTERVAL_SERVICE = 9.5 # Service takes roughly 9.5 seconds of time

ARR_RATE = 1/INTERVAL_CUSTOMERS  #lambda
SERV_RATE = 1/INTERVAL_SERVICE   #mu

# The script subsections, comment out for partial run.
if __name__ == '__main__':
    # Run simulations
    random.seed(RANDOM_SEED)
    es_list, wait_list = run_analytical_simulation(capacities)
    plot_analytic_result(es_list, wait_list, capacities)
    run_simulation(capacities, service_dis)

    # Process result data
    infile = open("data.pickle", 'rb')
    data = pkl.load(infile)
    plot_functions(data, capacities, metrics, service_dis)
    get_stats(data, capacities, metrics, service_dis)
    make_table("stats.pickle")
    do_stat_test("stats.pickle")

