import os
from os import path as ospath
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_allsess_paths(subject,proj_path):
    session_paths = []
    total_sess = 0

    for subj in subject:
        subj_path = os.path.join(proj_path, subj)
        items = os.listdir(subj_path)
        days = [item for item in items if item != '.DS_Store']

        print('Subject:', subj)

        for day in days:
            day_path = os.path.join(subj_path, day)
            sessions = [item for item in os.listdir(day_path) if item != '.DS_Store']

            print('Day:', day, 'Sessions:', len(sessions))

            total_sess += len(sessions)

            for session in sessions:
                session_path = os.path.join(day_path, session)
                session_paths.append(session_path)
    return session_paths
