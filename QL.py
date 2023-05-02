import argparse
import os
import droidbot_init
import json
import gym
import numpy as np
import sys
import glob

from check_file import get_max

def parse_args():
    parser = argparse.ArgumentParser(description="Start DroidBot to test an Android app.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-a", action="store", dest="apk_path", required=False, default='../com.smorgasbork.hotdeath_11.apk',
                        help="The file path to target APK")
    parser.add_argument("-o", action="store", dest="output_dir", default='output_test_1_1000/',
                        help="directory of output")
    parser.add_argument("-policy", action="store", dest="input_policy", default='gym')

    parser.add_argument("-distributed", action="store", dest="distributed", choices=["master", "worker"],
                        help="Start DroidBot in distributed mode.")
    parser.add_argument("-master", action="store", dest="master",
                        help="DroidMaster's RPC address")
    parser.add_argument("-qemu_hda", action="store", dest="qemu_hda",
                        help="The QEMU's hda image")
    parser.add_argument("-qemu_no_graphic", action="store_true", dest="qemu_no_graphic",
                        help="Run QEMU with -nograpihc parameter")

    parser.add_argument("-script", action="store", dest="script_path",
                        help="Use a script to customize input for certain states.")
    parser.add_argument("-count", action="store", dest="count", default=1000, type=int,
                        help="Number of events to generate in total. Default: %d" % 1000)
    parser.add_argument("-interval", action="store", dest="interval", default=1,
                        type=int,
                        help="Interval in seconds between each two events. Default: %d" % 1)
    parser.add_argument("-timeout", action="store", dest="timeout", default=-1, type=int,
                        help="Timeout in seconds, -1 means unlimited. Default: %d" % -1)
    parser.add_argument("-cv", action="store_true", dest="cv_mode",
                        help="Use OpenCV (instead of UIAutomator) to identify UI components. CV mode requires opencv-python installed.")
    parser.add_argument("-debug", action="store_true", dest="debug_mode",
                        help="Run in debug mode (dump debug messages).")
    parser.add_argument("-random", action="store_true", dest="random_input",
                        help="Add randomness to input events.")
    parser.add_argument("-keep_app", action="store_true", dest="keep_app",
                        help="Keep the app on the device after testing.")
    parser.add_argument("-keep_env", action="store_true", dest="keep_env",
                        help="Keep the test environment (eg. minicap and accessibility service) after testing.")
    parser.add_argument("-use_method_profiling", action="store", dest="profiling_method",
                        help="Record method trace for each event. can be \"full\" or a sampling rate.")
    parser.add_argument("-grant_perm", action="store_true", dest="grant_perm",
                        help="Grant all permissions while installing. Useful for Android 6.0+.")
    parser.add_argument("-is_emulator", action="store_true", dest="is_emulator",
                        help="Declare the target device to be an emulator, which would be treated specially by DroidBot.")
    parser.add_argument("-accessibility_auto", action="store_true", dest="enable_accessibility_hard",
                        help="Enable the accessibility service automatically even though it might require device restart\n(can be useful for Android API level < 23).")
    parser.add_argument("-humanoid", action="store", dest="humanoid",
                        help="Connect to a Humanoid service (addr:port) for more human-like behaviors.")
    parser.add_argument("-ignore_ad", action="store_true", dest="ignore_ad",
                        help="Ignore Ad views by checking resource_id.")
    parser.add_argument("-replay_output", action="store", dest="replay_output",
                        help="The droidbot output directory being replayed.")

    options = parser.parse_args()

    return options


def main():
    opts = parse_args()
    if not os.path.exists(opts.apk_path):
        print("APK does not exist.")
        return
    if not opts.output_dir and opts.cv_mode:
        print("To run in CV mode, you need to specify an output dir (using -o option).")



    state_function = {}
    num_iterations = 1000
    EPSILON = 0
    Q_TABLE = []
    #print("1")
    transitions_matrix = None
    number_of_trans = []
    #print("type at init :", type(number_of_trans))
    event_to_id = []
    max_number_of_actions = 50
    ALPHA = 0.9
    GAMMA = 0.02
    freq = []

    env = droidbot_init.start(opts)
    def events_so_state(env):
        events = env.envs[0].possible_events
        state_now = env.envs[0].device.get_current_state()
        event_ids = []
        probs = []

        for i, event in enumerate(events):
            event_str = str(type(event)) + '_' + event.get_event_str(state_now)
            if event_str in event_ids:
                1/0
            if event:
                event_ids.append(event_str)
                probs.append(env.envs[0].events_probs[i])
        state = state_now.state_str
        #print("state id :", state)
        #print("********************************************************************")
        #print("********************************************************************")
        probs = np.array(probs)
        #print("state : ", state)
        #print("state now: ", state_now)
        #print("event ids :", event_ids)
        return state, probs, event_ids, state_now
    
    '''    
    Q_TABLE = np.load('final_q_table5.npy')
    #print("q table shape :", Q_TABLE.shape)
    freq = np.load('freq5.npy')
    transitions_matrix = np.load('transition_function_5.npy')
    #print("transition matrix shape :", transitions_matrix.shape)
    transitions = np.load('transitions_5.npy')
    number_of_trans = np.load('number_of_trans_5.npy')
    number_of_trans = number_of_trans.tolist()
    good_states = np.load('good_states_5.npy')
    new_q_values = np.load('new_q_values_5.npy')
    q_target = np.load('q_target_5.npy')
    with open("states_5_test.json", "r") as f:
        state_function = json.loads(f.read())
    with open("event_to_id_5.json", "r") as f1:
        event_to_id = json.loads(f1.read())
    '''
    
    Q_TABLE = np.load('QTable_'+str(get_max('QTable_*_.npy'))+'_.npy')
    freq = np.load('freq_'+str(get_max('freq_*_.npy'))+'_.npy')
    transitions_matrix = np.load('transitionFunction_'+str(get_max('transitionFunction_*_.npy'))+'_.npy')
    number_of_trans = np.load('NumberOfTrans_'+str(get_max('NumberOfTrans_*_.npy'))+'_.npy').tolist()
    good_states = np.load('GoodStates_'+str(get_max('GoodStates_*_.npy'))+'_.npy')
    new_q_values = np.load('NewQValues_'+str(get_max('NewQValues_*_.npy'))+'_.npy')
    q_target = np.load('QTarget_'+str(get_max('QTarget_*_.npy'))+'_.npy')
    transitions = np.load('transitions_'+str(get_max('transitions_*_.npy'))+'_.npy')
    num_states = np.load('NumStates_'+str(get_max('NumStates_*_.npy'))+'_.npy')
    with open('states_'+str(get_max('states_*_test.json'))+'_test.json', 'r') as f1:
        state_function = json.loads(f1.read())
    with open('EventToID_'+str(get_max('EventToID_*_.json'))+'_.json', 'r') as f2:
        event_to_id = json.loads(f2.read())
    

    def check_state(state_id, events):
        nonlocal Q_TABLE
        nonlocal transitions_matrix
        nonlocal number_of_trans
        nonlocal event_to_id
        nonlocal state_function
        nonlocal freq

        #print("state i ", state_i)

        if state_function.get(state_id) is None:
            if Q_TABLE == []:
                #print("2")
                Q_TABLE = np.zeros((1, max_number_of_actions))
                freq = np.zeros((1, max_number_of_actions))
                transitions_matrix = np.zeros((1, max_number_of_actions, 1))
            else:
                #print("3")
                Q_TABLE = np.concatenate([Q_TABLE, np.zeros((1, max_number_of_actions))], axis=0)
                freq = np.concatenate([freq, np.zeros((1, max_number_of_actions))], axis=0)
                #print("4")
                transition_matrix_new = np.zeros((Q_TABLE.shape[0], max_number_of_actions, Q_TABLE.shape[0]))
                transition_matrix_new[:-1, :, :-1] = transitions_matrix
                transitions_matrix = transition_matrix_new
            event_to_id.append({})
            state_function[state_id] = Q_TABLE.shape[0] - 1
            Q_TABLE[-1][-1] = 50.0
            freq[-1][-1] = 1.0
            #print("5")
            number_of_trans.append(np.zeros(max_number_of_actions))
            #print("after init :", type(number_of_trans))
        state_i = state_function.get(state_id)
        #print("################################")
        #print("state_i", state_i)
        #print("###############################")
        id_to_action = np.zeros((max_number_of_actions), dtype=np.int32) + 1000
        q_values = np.zeros(max_number_of_actions)
        probs_now = np.zeros(max_number_of_actions)
        #print("events possible: ", events)
        for i, event in enumerate(events):
            if i == len(events) - 1:
                #print("12")
                q_values[-1] = Q_TABLE[state_i][-1]
                id_to_action[-1] = min(len(events), max_number_of_actions) - 1
                continue
            
            if event_to_id[state_i].get(event) is None:
                if len(event_to_id[state_i]) >= max_number_of_actions - 1:
                    continue
                event_to_id[state_i][event] = int(len(list(event_to_id[state_i].keys())))
                #print("6")
                Q_TABLE[state_i][event_to_id[state_i][event]] = 50.0
                freq[state_i][event_to_id[state_i][event]] = 1.0
                #print("7")
            q_values[event_to_id[state_i][event]] = Q_TABLE[state_i][event_to_id[state_i][event]]
            #print("8")
            id_to_action[event_to_id[state_i][event]] = int(i)
        #print("qtable in check state :", Q_TABLE)
        return id_to_action

    state_pre, probs, event_ids, state_now = events_so_state(env)
    #print("state pre : ", state_pre)
    id_to_action = check_state(state_pre, event_ids)
    state = state_function[state_pre]
    reward = 0
    #freq = np.zeros(1, max_number_of_actions)

    def make_decision(state_i, id_to_action):
        nonlocal Q_TABLE
        if np.random.rand() < EPSILON:
            action = max_number_of_actions - 1
            make_action = id_to_action[action]
        else:
            max_q = np.max(Q_TABLE[state_i])
            actions_argmax = np.arange(max_number_of_actions)[Q_TABLE[state_i] >= max_q - 0.0001]
            probs_unnormed = 1/(np.arange(actions_argmax.shape[0]) + 1.)
            probs_unnormed /= np.sum(probs_unnormed)
            action = np.random.choice(actions_argmax)
            make_action = id_to_action[action]
        #print("q table inside make decision :", Q_TABLE)
        return action, make_action

    def get_reward(state_index, state_now):
        nonlocal freq
        #print("app = ", env.envs[0].input_manager.app)
        if state_now.get_app_activity_depth(env.envs[0].input_manager.app) < 0:
            # If the app is not in the activity stack
            reward = -1000
            return reward
        #print("no of ones are ", Q_TABLE[state_index].tolist().count(1))
        reward = freq[state_index].tolist().count(1)
        return reward
    

    #print("state function :", state_function)
    

    for i_step in np.arange(num_iterations):
        action, make_action = make_decision(state, id_to_action)
        #print(Q_TABLE)
        #print(state, action, make_action)
        env.step([make_action])
        new_state_pre, probs, event_ids, state_now = events_so_state(env)

        id_to_action = check_state(new_state_pre, event_ids)
        new_state = state_function[new_state_pre]

        reward = get_reward(new_state, state_now)
        #print("reward :", reward)

        number_of_trans[state][action] += 1
        transitions_matrix[state, action] *= (number_of_trans[state][action] - 1)
        transitions_matrix[state, action, new_state] += 1
        transitions_matrix[state, action] /= number_of_trans[state][action]
        #for _ in np.arange(10):
        for i in np.arange(max_number_of_actions):
            transitions = transitions_matrix[:, i, :]
            #print("transition matrix shape :", transitions_matrix.shape)
            q_target = np.array([[np.max(Q_TABLE[i])] for i in np.arange(Q_TABLE.shape[0])])
            #print("q_target :", q_target)
            #print("transitions shape : ",transitions.shape)
            #print("q target shape : ", q_target.shape)
            new_q_values = np.matmul(transitions, q_target) * GAMMA * reward
            #print("new_q_values : ", new_q_values)
            good_states = np.sum(transitions, axis=1) > 0.5
            #print("good states :", good_states)
            if True in good_states:
                #print("10")
                #Q_TABLE[good_states, i] = new_q_values[good_states, 0]
                Q_TABLE[state, action] = (1-ALPHA) * Q_TABLE[state, action] + ALPHA * (reward * GAMMA * np.max(Q_TABLE[new_state, :]))

                #print("state :", state)
                #print("Action :" , action)
                if freq[state][action] == 1.0:
                    freq[state][action] = 0

                #print("q table in loop :", Q_TABLE)
            else:
                continue
        #print("11")
        #print("q table outside loop : ", Q_TABLE)
        #for i in np.arange(Q_TABLE.shape[0]):
            #print("Q_TABLE")
        #if i_step%10==0:
        #np.save('q_function1', Q_TABLE)
        #print("transition matrix shape :", transitions_matrix.shape)
        #np.save('transition_function_1', transitions_matrix)
        #with open('states_6_test.json', 'w') as f:
        #    json.dump(state_function, f)
        #with open('event_to_id_6.json', 'w') as f1:
        #    json.dump(event_to_id, f1)
        #with open('number_of_trans_1.json', 'w') as f2:
        #    json.dump(number_of_trans, f2)
        #np.save('new_q_values_6', new_q_values)
        #np.save('number_of_trans_6', number_of_trans)
        #np.save('good_states_6', good_states)
        #np.save('q_target_6', q_target)
        #np.save('transitions_6', transitions)
        #np.save('transitionFunction_'+str(get_max('transitionFunction_*_.npy')+1)+'_.npy')

        state = new_state
    with open('states_'+str(get_max('states_*_test.json')+1)+'_test.json', 'w') as f3:
        json.dump(state_function, f3)
    with open('EventToID_'+str(get_max('EventToID_*_.json')+1)+'_.json', 'w') as f4:
        json.dump(event_to_id, f4)
    num_states = Q_TABLE.shape[0]
    print("number of states :", num_states)
    np.save('NumStates_'+str(get_max('NumStates_*_.npy')+1)+'_.npy', num_states)
    np.save('NewQValues_'+str(get_max('NewQValues_*_.npy')+1)+'_.npy', new_q_values)
    np.save('NumberOfTrans_'+str(get_max('NumberOfTrans_*_.npy')+1)+'_.npy', number_of_trans)
    np.save('GoodStates_'+str(get_max('GoodStates_*_.npy')+1)+'_.npy', good_states)
    np.save('QTarget_'+str(get_max('QTarget_*_.npy')+1)+'_.npy', q_target)
    np.save('transitions_'+str(get_max('transitions_*_.npy')+1)+'_.npy', transitions)
    #print("transition matrix shape :", transitions_matrix.shape)
    np.save('transitionFunction_'+str(get_max('transitionFunction_*_.npy')+1)+'_.npy', transitions_matrix)
    #print("Q_table shape :", Q_TABLE.shape)
    np.save('QTable_'+str(get_max('QTable_*_.npy')+1)+'_.npy', Q_TABLE)
    np.save('freq_'+str(get_max('freq_*_.npy')+1)+'_.npy', freq)
    #print(Q_TABLE)
    #1/0
    #droidbot.stop()
    print("end")

if __name__ == "__main__":
    print("Starting Q Learning")
    main()
