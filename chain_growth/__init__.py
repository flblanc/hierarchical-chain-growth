#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

# MD fragment PDBs throughout this pipeline are minimal (no chainIDs, unit cell,
# formal charges, or trajectory dt) by construction -- MDAnalysis's own UserWarnings
# about these are expected noise here, not something users need to see on every run.
warnings.filterwarnings('ignore', category=UserWarning, module='MDAnalysis')
