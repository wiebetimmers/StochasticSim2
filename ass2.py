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
import simpy
import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,4), dpi=300)


RANDOM_SEED = 42
NEW_CUSTOMERS = 10000  # Total number of customers
INTERVAL_CUSTOMERS = 12.0  # Generate new customers roughly every x seconds
INTERVAL_SERVICE = 10.0 # Service takes roughly 10 seconds of time

ARR_RATE = 1/INTERVAL_CUSTOMERS
SERV_RATE = 1/ INTERVAL_SERVICE
CAP = 1


def source(env, number, counter):
    soj_times = []
    """Source generates customers randomly"""
    for i in tqdm(range(number)):
        c = customer(env, 'Customer%02d' % i, counter)
        soj_time = yield env.process(c)
        soj_times.append(soj_time)
        t = random.expovariate(ARR_RATE)
        yield env.timeout(t)

    print('Mean soj time: ', st.mean(soj_times))
    #sns.displot(soj_times)
    #plt.xlabel('sojourn time')
    #plt.savefig('sojourn_displot_%s.jpg'%CAP)

def customer(env, name, counter):
    """Customer arrives, is served and leaves."""
    arrive = env.now
    #print('%7.4f %s: Here I am' % (arrive, name))

    with counter.request() as req:
        results = yield req | env.timeout(1000)

        wait = env.now - arrive

        if req in results:
            # We got to the counter
            #print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
            # We get served
            tib = random.expovariate(SERV_RATE)
            yield env.timeout(tib)
            end = env.now
            soj_time = end-arrive
            return soj_time
            #print('%7.4f %s: Sojourn Time' % (end-arrive, name))
            #print('%7.4f %s: Finished' % (env.now, name))


# Setup and start the simulation
random.seed(RANDOM_SEED)
env = simpy.Environment()

# Start processes and run
capacities = list(range(1,11))
es_list = []
for cap in capacities:
    CAP = cap
    print('Bank renege')
    system_load = ARR_RATE / (CAP * SERV_RATE)
    el = (system_load)/(1-system_load)
    es = (1/SERV_RATE)/(1-system_load)
    print('System load: ', system_load)
    es_list.append(es)
    counter = simpy.Resource(env, capacity=CAP)
    env.process(source(env, NEW_CUSTOMERS, counter))
    env.run()

plt.plot(capacities, es_list)
plt.xlabel('Capacity')
plt.ylabel('E(s)')
plt.savefig('expdecay_sojourningtime.jpg')