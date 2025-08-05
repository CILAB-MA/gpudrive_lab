import os
import sys
import json
import argparse
import numpy as np

from tqdm import tqdm

import numpy as np
import os
import re
from collections import defaultdict
import random

def group_answers_by_unique_question(int_qa):
    q_to_a = defaultdict(list)
    for q, a in int_qa:
        q_to_a[q].append(a)
    return q_to_a

def extract_agent_numbers(sentence):
    agent_numbers = set()
    matches = re.findall(r"#(\d+)", sentence)
    agent_numbers.update(matches)
    return sorted(agent_numbers, key=int)

def simple_remap_ids(texts, original_ids):
    mapping = {oid: str(random.randint(0, 15)) for oid in original_ids}
    remapped = []
    for text in texts:
        for oid in original_ids:
            text = text.replace(f"#{oid}", f"#{mapping[oid]}")
        remapped.append(text)
    return remapped, mapping

def generate_all_neg(questions, answers, qa_type='int'):
    neg_answers = []
    for question, answer in zip(questions, answers):
        if qa_type == 'ego':
            neg_answer = generate_negative_ego(answer)
        if qa_type == 'int':
            neg_answer = generate_negative_int(question, answer)
        if qa_type == 'sur':
            neg_answer = generate_negative_sur(question, answer)
        if qa_type == 'env':
            neg_answer = generate_negative_env(question, answer)
        neg_answers.append(neg_answer)
        # if 'only lane' in answer:
        #     print(answer)
    neg_answers = np.array(neg_answers)
    answer_pairs = np.concatenate([answers.reshape(-1, 1), neg_answers.reshape(-1, 1)], axis=-1)
    return answer_pairs

def generate_negative_env(question, answer):
    neg_answer = answer.copy()
    change = False
    done_change = False
    q_target = extract_agent_numbers(question)
    a_target = extract_agent_numbers(answer)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer)
    if len(numbers) > 0:
        change = True
        orig_number = [int(eval(n)) for n in numbers]
        max_range = max(orig_number) + 2 if max(orig_number) > 1 else 10
        candidate = np.arange(0, max_range)
        candidate = candidate[~np.isin(candidate, orig_number)]
        change_speed = np.random.choice(candidate, len(orig_number))
        for orig, new in zip(numbers, change_speed):
            neg_answer = neg_answer.replace(orig, str(new))

    template_patterns = [
        (r"No ([a-z ]+?)s? are present\.",
         lambda obj: f"{obj.capitalize()}s are clearly marked near the intersection."),
        (r"No ([a-z ]+?)s? are mentioned(?:.*)?\.",
         lambda obj: f"{obj.capitalize()}s are visible in the current scenario."),
        (r"No ([a-z ]+?)s? are indicated\.",
         lambda obj: f"{obj.capitalize()}s are indicated clearly."),
        (r"No ([a-z ]+?)s? are on the ego agent's side\.",
         lambda obj: f"{obj.capitalize()}s are directly on the ego agent's side."),
        (r"No ([a-z ]+?)s? are in the vicinity(?:.*)?\.",
         lambda obj: f"{obj.capitalize()}s are present in the vicinity."),
    ]
    for pattern, replacer in template_patterns:
        match = re.search(pattern, neg_answer)
        if match:
            obj = match.group(1).strip()
            neg_answer = replacer(obj)
            change = True
            done_change = True
            break

    replacements = [
        ('ahead of', 'behind'),
        ('not ', ''),
        ('four', str(np.random.randint(0, 4))),
        ('one ', f'{np.random.randint(2, 6)} '),
        ('three', f'{np.random.randint(4, 8)}'),
        ('seven', f'{np.random.randint(2, 6)}'),
        ('six', f'{np.random.randint(1, 5)}'),
        ('five', f'{np.random.randint(0, 4)}'),
        ('eight', f'{np.random.randint(4, 8)}'),
        ('nine', f'{np.random.randint(4, 8)}'),
        ('No,', 'Yes,'),
        ('No.', 'Yes.'),
        ('Yes.', 'No.'),
        (' no ', ' '),
        ('Yes,', 'No,'),
        ('within', 'outside'),
        ('there is a ', 'ther is no '),
        ('is at', 'is not at'),
        ('stationary', 'moving'),
        ('is a ', 'is no '),
        ('directly', 'not'),
        ('opposite ', 'same '),
        ("No stop signs, crosswalks, or speed bumps are mentioned",
         "Stop signs, crosswalks, and speed bumps are all present"),
        ('located at', 'outside'),
        ('approaching', np.random.choice(['already in', 'departing from'])),
        ('only lane', f"{np.random.randint(3)} lane from the {np.random.choice(['left', 'right'])}"),
        ("an entry or exit intersection",
         "a through intersection that does not serve as an entry or exit"),
        ("No speed bumps are mentioned.", "There are speed bumps behind."),
        ('two', f'{np.random.randint(3, 7)}'),
        ('constant speed', np.random.choice(['accelerating', 'decelerating'])),
        ('accelerating', 'decelerating'),
        ('decelerating', 'accelerating'),
        ('same ', np.random.choice(['opposite ', 'differnt '])),
        ('is in', 'is exiting'),
        ('left', 'right'),
        ('right', 'left'),
        ("One", "Three"),
        ("Three lanes", "No lanes"),
        ('present', 'traveling'),
        ('multiple', 'no'),
        ('entering', 'leaving'),
        ('heading towards', 'moving away from'),
        ('behind', 'in front of'),
        ('going straight', np.random.choice(['turning left', 'turning right'])),
        ('exiting', 'approaching'),
        ("There are crosswalks present.",
         "No crosswalks are visible."),
        ("various", "constant"),
        ("are in the", "are far from"),
        ("near ", "far from "),
        ('Vehicles are moving on the road.', 'Vehicles are stopping on the road.'),
        ("traveling", "stationary"),
        ("There are stop signs and a crosswalk.",
         "There are no stop signs or crosswalks present."),
        ("There are stop signs and a speed bump.",
         "There are no stop signs or speed bumps in the area."),
        ("No speed bumps are being approached by the ego agent.",
         "The ego agent is about to cross a speed bump."),
        ("Traffic is flowing with vehicles at various speeds.",
         "There is no visible traffic flow in the current scenario."),
        ("Vehicles are moving in both directions on the road.",
         "Vehicles are only moving in one direction on the road."),
        ("There is an intersection.",
         "There is no intersection in the current view."),
        ('The intersection layout affects the driving scene by providing a point where the ego agent and surrounding agents can enter or exit.',
         'The intersection layout plays no role in enabling entry or exit for the ego agent or surrounding agents in the driving scene.'),
        ("The intersection layout affects the driving scenario by providing entry or exit points for the vehicles.",
         "The intersection layout does not influence vehicle behavior."),
        ("The intersection layout affects the driving conditions by providing entry or exit paths for the ego agent.",
         "The layout has minimal impact on the ego agent's driving behavior."),
        ("on a single lane.",
         "in a multi-lane road segment."),
        ("The stop sign is for agents other than the ego agent.",
         "The stop sign is specifically for the ego agent."),
        ("The stop sign applies to the direction the ego agent is facing.",
         "The stop sign does not apply to the ego agent's direction."),
        ("No stop signs are affecting the current driving environment.",
         "A stop sign is currently active and relevant to the ego agent."),
        ("The stop signs are at the intersection.",
         "There are no stop signs visible at the intersection."),
        ("The ego agent is departing from an intersection.",
         "The ego agent is entering the intersection."),
        ("constant at the moment", f"{np.random.choice(['accelerating', 'decelerating'])} at the moment"),
        ("No stop sign is on the ego agent's side.",
         "There is a stop sign directly on the ego agent's side."),
        ("The stop sign is for the ego agent's direction.",
         "The stop sign is not intended for the ego agent."),
        ("Vehicles are driving on the road.",
         "No vehicles are currently driving on the road."),
        ("No crosswalk is mentioned", "A crosswalk is clearly visible"),
        ("No crosswalks are present", "Crosswalks are clearly marked near the intersection"),
    ]
    if not done_change:
        for orig, new in replacements:
            if orig in answer:
                change = True
                neg_answer = neg_answer.replace(orig, new)
    if answer == 'Yes':
        change = True
        neg_answer = 'No.'
    if answer == 'No':
        change = True
        neg_answer = 'Yes.'
    if not change:
        neg_answer = None
    return neg_answer

def generate_negative_int(question, answer):
    neg_answer = answer.copy()
    change = False
    q_target = extract_agent_numbers(question)
    a_target = extract_agent_numbers(answer)
    if q_target == [] and len(a_target) > 0:
        mapping = {oid: str(random.randint(0, 10)) for oid in a_target}
        for oid, nid in mapping.items():
            neg_answer = neg_answer.replace(f"#{oid}", f"#{nid}")
    replacements = [
    ('yield', 'not yield'),
    ('constant speed', np.random.choice(['accelerating', 'decelerating'])),
    ('maintain', np.random.choice(['be closer', 'be further'])),
    ('already in', np.random.choice(['approaching', 'departing from'])),
    ('approaching', np.random.choice(['already in', 'departing from'])),
    ('departing from', np.random.choice(['already in', 'approaching'])),
    ('left', 'right'),
    ('faster', 'slower'),
    ('lead ', 'follow '),
    ('be overtaken by', 'stay ahead of'),
    ('right', 'left'),
    ('stationary', 'moving'),
    ('behind', 'in front of'),
    ('is at', 'is not at'),
    ('not ', ''),
    (' no ', ' '),
    ('without any', 'with potential'),
    ('exit', 'enter'),
    ('opposite ', 'same '),
    ('same ', np.random.choice(['opposite ', 'differnt '])),
    ('different', 'same'),
    (' pass ', ' block '),
    (' stop ', ' continue '),
    ('passed by', 'block'),
    ("will be overtaking", "will not overtake"),
    ("moving from the side to the front of the ego agent", "remaining behind"),
    ("continue to decelerate", "accelerate"),
    ("remain in front", "fall behind"),
    (' towards', ' away from'),
    ('remains stationary', 'starts moving'),
    ('accelerating', 'decelerating'),
    ('decelerating', 'accelerating'),
    ('intends to continue', 'does not intend to continue'),
    ('managing its speed to navigate', 'stop before'),
    ('depart from', 'enter'),
    ('slow down', 'speed up'),
    ('slightly', 'significantly'),
    ('slower', 'faster'),
    ("stay still", "move into the ego agent's path"),
    ('continue towards', 'avoid'),
    ('follow', 'overtake'),
    ('overtake', 'follow'),
    ('high', 'slow'),
    ('adjacent', 'non-adjacent'),
    ('due to', 'despite'),
    ('ahead of', 'behind'),
    (' away from', ' towards'),
    ("managing its speed to navigate", "stop before"),
    ('No.', 'Yes.'),
    ('acceleration', 'deceleration'),
    ('continues to move forward', 'slows down to avoid collision'),
    ('to the side', 'in front of and in the path of'),
    ('increasing', 'decreasing'),
    ('decreasing', 'increasing'),
    ("in close proximity to", "far from"),
    ("suggesting a potential interaction", "indicating no interaction is expected"),
    ("managing its speed to navigate", "stop before"),
    ]
    for orig, new in replacements:
        if orig in answer:
            change = True
            neg_answer = neg_answer.replace(orig, new)
    if answer == 'Yes':
        change = True
        neg_answer = 'No.'
    if answer == 'No':
        change = True
        neg_answer = 'Yes.'

    if not change:
        neg_answer = None
    return neg_answer

def generate_negative_sur(question, answer):
    neg_answer = answer.copy()
    change = False
    # change the target id
    q_target = extract_agent_numbers(question)
    a_target = extract_agent_numbers(answer)
    if q_target == [] and len(a_target) > 0:
        mapping = {oid: str(random.randint(0, 10)) for oid in a_target}
        for oid, nid in mapping.items():
            neg_answer = neg_answer.replace(f"#{oid}", f"#{nid}")

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer)
    if len(numbers) > 0:
        change = True
        orig_number = [int(eval(n)) for n in numbers]
        max_range = max(orig_number) + 2 if max(orig_number) > 1 else 10
        candidate = np.arange(0, max_range)
        candidate = candidate[~np.isin(candidate, orig_number)]
        change_speed = np.random.choice(candidate, len(orig_number))
        for orig, new in zip(numbers, change_speed):
            neg_answer = neg_answer.replace(orig, str(new))

    replacements = [
        ('is a', 'is not a'),
        ('Vehicle', 'Not Vehicle'),
        ('A vehicle', 'Not a vehicle'),
        ('from the intersection center', 'from the ego agent' + np.random.choice(['', 'and in the intersection'])),
        ('from the ego agent', 'from the intersection center'),
        ('left', 'right'),
        (' no ', ' '),
        ('slightly', 'significantly'),
        ('same ', np.random.choice(['opposite ', 'differnt '])),
        ('opposite', 'same'),
        ('right', 'left'),
        ('in front of', 'behind'),
        ('behind', 'ahead'), # ahead of or in front of
        ('accelerating', 'decelerating'),
        ('decelerating', 'accelerating'),
        ('not ', ''),
        ('stationary', 'moving'),
        ('constant speed', np.random.choice(['accelerating', 'decelerating'])),
        ('approaching', 'departing from'),
        ('departing from', 'approaching'),
        ('towards', 'away from'),
        ('is in', 'is exiting'),
        ('No,', 'Yes,'),
        ('No.', 'Yes.'),
        ("No surrounding agents are mentioned",
         f"Surrounding agent #{np.random.choice(5)} is moving in front of the ego agent"),
        ('Yes.', 'No.'),
        ('constant.', np.random.choice(['accelerating.', 'decelerating.'])),
        ('Left', 'Right'),
        ('Right', 'Left'),
        ("No crosswalk is mentioned.", "A crosswalk is clearly visible."),
        ("No speed bump is mentioned.", "There is a speed bump in front."),
        ("No speed bumps are mentioned.", "There are speed bumps behind."),
        ("No stop sign is mentioned.", "A stop sign is located at the intersection."),
        ("It is currently at the stop sign.", "It has already passed the stop sign."),
        ("It is facing away from the intersection.", "It is facing towards the intersection."),
        ("Yes, there are stop signs.", "No, there are no stop signs."),
        ("Yes, there are multiple surrounding agents", "No, there are no surrounding agents nearby."),
        ("No speed bump is mentioned for the surrounding agents.",
         "A speed bump is located near a surrounding agent."),
        ("No stop signs are mentioned for the surrounding agents.",
         "There is a stop sign for one of the surrounding agents."),
        ("No crosswalks are mentioned for the surrounding agents.",
         "A crosswalk is mentioned in relation to a surrounding agent."),
        ("No crosswalk or speed bump is mentioned for either agent.",
         "At least one agent is associated with a crosswalk or speed bump."),
        ("Yes, a crosswalk", "No crosswalk is mentioned."),
    ]
    for orig, new in replacements:
        if orig in answer:
            change = True
            neg_answer = neg_answer.replace(orig, new)
    if answer == 'No':
        change = True
        neg_answer = 'Yes.'
    if answer == 'Yes':
        change = True
        neg_answer = 'No.'
    if not change:
        neg_answer = None
    return neg_answer

def generate_negative_ego(answer):
    neg_answer = answer.copy()
    change = False

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer)
    if len(numbers) > 0:
        change = True
        orig_number = [int(eval(n)) for n in numbers]
        max_range = max(orig_number) + 1 if max(orig_number) > 1 else 10
        candidate = np.arange(0, max_range)
        candidate = candidate[~np.isin(candidate, orig_number)]
        change_speed = np.random.choice(candidate, len(orig_number))
        for orig, new in zip(numbers, change_speed):
            neg_answer = neg_answer.replace(orig, str(new))

    replacements = [
        ('left', 'right'),
        ('right', 'left'),
        ('ahead', 'behind'),
        ('behind', 'ahead'),
        ('accelerating', 'decelerating'),
        ('decelerating', 'accelerating'),
        ('constant speed', np.random.choice(['accelerating', 'decelerating'])),
        ('No', 'Yes'),
        ('Yes', 'No'),
        ('is at', "isn't at"),
        ('going straight', np.random.choice(['turning left', 'turning right'])),
        ('approaching', 'departing from'),
        ('departing from', 'approaching'),
        ('only lane', f"{np.random.randint(3)} lane from the {np.random.choice(['left', 'right'])}"),
        ('no stop sign', '1 stop sign'),
        ('exiting', 'approaching'),
        ('is in', 'is exiting'),
        ('not ', ''),  # remove negation
        ('not specified', 'clearly specified'),
        ('no crosswalk', 'a crosswalk'),
        ('no speed bump', 'a speed bump'),
        ('no mention', 'a mention'),
        ('not influenced by', 'is influenced by'),
        ('away from', 'towards'),
        ('towards', 'away from'),
        ('making a U-turn', 'not making a U-turn'),
        ('heading in the same direction', 'heading in a different direction'),
        ('not making any turns', 'making a turn'),
        ('facing forward in', 'facing away from'),
        ('moving forward', 'reversing'),
        ('moving straight', 'turning'),
        ('facing straight', 'facing in a different direction'),
        ('entering the intersection', 'exiting the intersection'),
        ('facing in the direction', 'facing in the opposite direction'),
        ('on the same lane', 'on a different lane'),
        ('There is no information about', 'There is information about'),
        ('only one lane', 'multiple lanes'),
        ('four lanes on', 'no lanes on'),
        ('decreasing', 'increasing'),
        ('increasing', 'decreasing'),
        ('will encounter a speed bump', 'will not encounter a speed bump'),
        ('maintaining its speed', 'changing its speed'),
    ]
    for orig, new in replacements:
        if orig in answer:
            change = True
            neg_answer = neg_answer.replace(orig, new)

    constant_phrases = [
        "The ego agent's speed is constant at the moment",
        "The ego agent's speed is constant"
    ]
    for phrase in constant_phrases:
        if phrase in answer:
            change = True
            replacement = f"The ego agent is {np.random.choice(['accelerating', 'decelerating'])} at the moment"
            neg_answer = neg_answer.replace(phrase, replacement)

    direction_phrases = {
        "heading in its current direction": "heading in the opposite direction",
        "heading in the direction of the lane it is on": "heading in the opposite direction of the lane",
        "heading in its current travel direction": "heading in the reverse direction",
        "There is no specific facing direction mentioned for the ego agent": "The ego agent is clearly facing forward",
        "facing the same direction as the current moment": "facing the opposite direction as the current moment",
        "facing the direction it is traveling": "facing away from the direction it is traveling",
        "heading in a straight direction": np.random.choice(["turning left", "turning right"]), 
        "about to encounter a speed bump": "not about to encounter a speed bump", 
        "heading in the direction it is currently traveling": "heading in the opposite direction it was traveling",
        "heading in the direction of its lane": "heading opposite to the lane direction",
        "heading in its current direction of travel": "heading in the opposite direction of travel",
        "moving in the same direction as its current lane": "moving in the opposite direction of its current lane",
        "moving in the same direction as its lane": "moving in the reverse direction of its lane",
        "moving in the same direction as its current moment": "moving in the opposite direction of its current moment",
        "facing the direction of travel": "facing away from the direction of travel",
        "facing in the same direction as its current movement": "facing in the opposite direction of its movement",
        "facing in the same direction as it was at the current moment": "facing in the opposite direction as it was at the current moment",
        "facing the same direction it was initially heading": "facing the reverse direction from initial heading",
        "facing the direction of the lane it is on": "facing away from the direction of the lane it is on",
        "moving in the same direction as the surrounding agents": "moving in the opposite direction of the surrounding agents",
        "heading straight": np.random.choice(["turning left", "turning right"]),
        "moving in a straight direction": "moving in a curved direction",
        "traveling straight": np.random.choice(["turning", "reversing"]),
        "on the same side of the intersection": "on the opposite side of the intersection"
    }
    for phrase, replacement in direction_phrases.items():
        if phrase in answer:
            change = True
            neg_answer = neg_answer.replace(phrase, replacement)

    if answer.strip().lower() == "left.":
        change = True
        neg_answer = "Right."
    elif answer.strip().lower() == "right.":
        change = True
        neg_answer = "Left."
    elif answer.strip().lower() == "decelerating":
        change = True
        neg_answer = "Accelerating"
    elif answer.strip().lower() == "accelerating":
        change = True
        neg_answer = "Decelerating"

    info_negations = {
        "There is no specific information about the ego agent's current lane": 
            "There is specific information about the ego agent's current lane",
        "There is no information on the ego agent's facing direction": 
            "There is clear information about the ego agent's facing direction",
        "There is no information provided about the ego agent's facing direction": 
            "The ego agent's facing direction is clearly provided",
        "There is no information on the ego agent's turning direction": 
            "The ego agent's turning direction is clearly indicated",
        "There is no information on the ego agent making any turns": 
            "The ego agent is making a turn",
        "There are no traffic control devices affecting the ego agent": 
            "There are traffic control devices affecting the ego agent",
        "There are no traffic controls mentioned affecting the ego agent's movement": 
            "There are traffic controls mentioned affecting the ego agent's movement"
    }
    for phrase, replacement in info_negations.items():
        if phrase in answer:
            change = True
            neg_answer = neg_answer.replace(phrase, replacement)

    if not change:
        neg_answer = None
    return neg_answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="training", help="training (80000) / testing (10000)")
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument("--total-scene-size", "-tss", type=int, default=80000)
    parser.add_argument("--scene-batch-size", "-sbs", type=int, default=50)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=128)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()
    TOTAL_NUM_WORLDS = args.total_scene_size
    NUM_WORLDS = args.scene_batch_size
    num_iter = int(TOTAL_NUM_WORLDS // NUM_WORLDS)
    env_qs = []
    ego_qs = []
    sur_qs = []
    int_qs = []
    env_as = []
    ego_as = []
    sur_as = []
    int_as = []
    int_qas = []
    sur_qas = []
    env_qas = []
    ego_qas = []
    qa_types = ["env", "ego", "sur", "int"]
    for idx in tqdm(range(num_iter)):
        if idx != num_iter - 1:
            with open(f"/data/full_version/processed/reasoning_raw/{args.data_dir}/womd_reasoning_{100 * idx}.json", "r") as f:
                jd = json.load(f)
            valid_agent = len(jd)
            expert_env_q_lst = np.empty((valid_agent, 20, ), dtype=object)
            expert_ego_q_lst = np.empty((valid_agent, 20, ), dtype=object)
            expert_sur_q_lst = np.empty((valid_agent, 120, ), dtype=object)
            expert_int_q_lst = np.empty((valid_agent, 30, ), dtype=object)

            expert_env_ap_lst = np.empty((valid_agent, 20, ), dtype=object)
            expert_ego_ap_lst = np.empty((valid_agent, 20, ), dtype=object)
            expert_sur_ap_lst = np.empty((valid_agent, 120, ), dtype=object)
            expert_int_ap_lst = np.empty((valid_agent, 30, ), dtype=object)

            expert_env_an_lst = np.empty((valid_agent, 20, ), dtype=object)
            expert_ego_an_lst = np.empty((valid_agent, 20, ), dtype=object)
            expert_sur_an_lst = np.empty((valid_agent, 120, ), dtype=object)
            expert_int_an_lst = np.empty((valid_agent, 30, ), dtype=object)
            q_npy = [expert_env_q_lst, expert_ego_q_lst, expert_sur_q_lst, expert_int_q_lst]
            ap_npy = [expert_env_ap_lst, expert_ego_ap_lst, expert_sur_ap_lst, expert_int_ap_lst]
            an_npy = [expert_env_an_lst, expert_ego_an_lst, expert_sur_an_lst, expert_int_an_lst]

            for i, data in enumerate(jd.values()):
                for qa, qa_type in enumerate(qa_types):
                    qas = data[f"{qa_type}_qa"]
                    if qas:
                        qs, ans = zip(*qas) 
                    else:
                        continue
                    answer_pairs = generate_all_neg(np.array(qs), np.array(ans), qa_type=qa_type)
                    qs = np.array(qs)
                    pos_answer = answer_pairs[:, 0]
                    neg_answer = answer_pairs[:, 1]
                    num = min(len(qs), q_npy[qa].shape[1])
                    q_npy[qa][i, :num] = qs[:num]
                    ap_npy[qa][i, :num] = pos_answer[:num]
                    an_npy[qa][i, :num] = neg_answer[:num]

            expert_env_q_lst = q_npy[0]
            expert_ego_q_lst = q_npy[1]
            expert_sur_q_lst = q_npy[2]
            expert_int_q_lst = q_npy[3]

            expert_env_ap_lst = ap_npy[0]
            expert_ego_ap_lst = ap_npy[1]
            expert_sur_ap_lst = ap_npy[2]
            expert_int_ap_lst = ap_npy[3]

            expert_env_an_lst = an_npy[0]
            expert_ego_an_lst = an_npy[1]
            expert_sur_an_lst = an_npy[2]
            expert_int_an_lst = an_npy[3]

            save_path = f'/data/full_version/processed/final/reasoning_{args.data_dir}_subset'
            os.makedirs(save_path + '/reasoning_posneg/nlp', exist_ok=True)
            np.savez_compressed(f"{save_path}/reasoning_posneg/nlp/reasoning_trajectory_{idx * args.scene_batch_size}.npz", 
                    env_q=expert_env_q_lst,
                    ego_q=expert_ego_q_lst,
                    sur_q=expert_sur_q_lst,
                    int_q=expert_int_q_lst,
                    env_pos_a=expert_env_ap_lst,
                    ego_pos_a=expert_ego_ap_lst,
                    sur_pos_a=expert_sur_ap_lst,
                    int_pos_a=expert_int_ap_lst,
                    env_neg_a=expert_env_an_lst,
                    ego_neg_a=expert_ego_an_lst,
                    sur_neg_a=expert_sur_an_lst,
                    int_neg_a=expert_int_an_lst,
                    )