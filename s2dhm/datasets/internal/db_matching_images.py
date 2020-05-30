# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

import sqlite3
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .db_handling import blob_to_array


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2


def get_matching_images(database_file, min_num_matches, filter_image_dir):
    connection = sqlite3.connect(database_file)
    cursor = connection.cursor()

    cursor.execute('SELECT image_id, name FROM images;')
    images = {image_id: name for image_id, name in cursor}

    two_way_matches = defaultdict(list)
    cursor.execute(
        'SELECT pair_id, data FROM two_view_geometries WHERE rows>=?;',
        (min_num_matches,))
    for pair_id, data in cursor:
        inlier_matches = blob_to_array(data, np.uint32, shape=(-1, 2))

        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        image_name1 = images[image_id1]
        image_name2 = images[image_id2]
        num_matches = inlier_matches.shape[0]

        # Make sure the match comes from the desired directory.
        if (image_name1.startswith(filter_image_dir)
                and image_name2.startswith(filter_image_dir)):
            two_way_matches[image_id1].append((image_id2, num_matches))
            two_way_matches[image_id2].append((image_id1, num_matches))

    matching_image_pairs = []
    for image_id, direct_matching_frames in tqdm(two_way_matches.items()):
        image_name = images[image_id]

        matching_frames = set()
        for matching_frame in direct_matching_frames:
            assert matching_frame[1] >= min_num_matches
            if matching_frame[0] > image_id:
                matching_frames.add(matching_frame[0])

        # Insert the direct matching pairs.
        for match in matching_frames:
            assert match > image_id
            matching_image_pairs.append((image_name, images[match]))

    cursor.close()
    connection.close()
    return matching_image_pairs
