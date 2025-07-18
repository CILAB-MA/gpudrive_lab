import os
import sys
import json
import argparse
import mediapy as media
import numpy as np

from tqdm import tqdm

import torch
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
        neg_answers.append(neg_answer)
        # if 'only lane' in answer:
        #     print(answer)
    neg_answers = np.array(neg_answers)
    answer_pairs = np.concatenate([answers.reshape(-1, 1), neg_answers.reshape(-1, 1)], axis=-1)
    return answer_pairs

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
    ('ahead of', 'behind'),
    ("in close proximity to", "far from"),
    ("suggesting a potential interaction", "indicating no interaction is expected"),
    ("managing its speed to navigate", "stop before"),
    ]
    for orig, new in replacements:
        if orig in answer:
            change = True
            neg_answer = neg_answer.replace(orig, new)
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
        ('going straight', np.random.choice(['turn left', 'turn right'])),
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
    for idx in range(num_iter):
        if idx != num_iter - 1:
            with open(f"/data/full_version/processed/reasoning_raw/{args.data_dir}/womd_reasoning_{100 * idx}.json", "r") as f:
                jd = json.load(f)
            for d in jd.values():
                env_qa = np.array(d['env_qa'])
                ego_qa = np.array(d['ego_qa'])
                sur_qa = np.array(d['sur_qa'])
                int_qa = np.array(d['int_qa'])
                if len(int_qa) > 0:
                    int_qas.append(int_qa)
                # print('Q:',int_qa[:, 0])
                # print('A:', int_qa[:, 1])
                # answer_pairs = generate_all_neg(ego_qa[:, 0], ego_qa[:, 1], qa_type='ego')
                if len(int_qa) > 0:
                    answer_pairs = generate_all_neg(int_qa[:, 0], int_qa[:, 1], qa_type='int')
                generation_ratio = (answer_pairs[:, 1] != None).sum() / len(answer_pairs)
                # print(f"Negative Generation Ratio: {generation_ratio:.4f}")
                # print('A:',answer_pairs)
                none_mask = answer_pairs[:, 1] == None
                none_negative = answer_pairs[none_mask]
                if len(none_negative) > 0:
                    print('A:', none_negative)
    # int_qas = np.concatenate(int_qas, axis=0)
    # q_to_a = group_answers_by_unique_question(int_qas)
    # for q, a in q_to_a.items():
    #     print(f'{q}: {a}')
    #     print('---------------------------------------------------------------------------')