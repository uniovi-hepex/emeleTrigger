# Assuming an 8*8 matrix of leds

hit_threshold = 3
axis = [elem / 8. for elem in range(8)]

min_slope = -1000
max_slope = +1000
d_slopes = [1]

all_straight_tracks = []

for d_slope in d_slopes:
    tracks = []
    for slope in range(min_slope, max_slope, d_slope):
        for y0 in axis:
            track = []
            for ix, x in enumerate(axis):
                y = (slope / 100.) * x + y0
                found = False
                for iy in range(len(axis[:-1])):
                    if y >= axis[iy] and y < axis[iy + 1]:
                        track.append((ix, iy))
                        break
                if not found:
                    if y >= axis[-1] and y < 1:
                        track.append((ix, len(axis) - 1))
            if slope != 0.:
                for iy, y in enumerate(axis):
                    x = (y - y0) / (slope / 100.)
                    found = False
                    for ix in range(len(axis[:-1])):
                        if x >= axis[ix] and x < axis[ix + 1]:
                            if (ix, iy) not in track:
                                track.append((ix, iy))
                            break
                    if not found:
                        if x >= axis[-1] and x < 1:
                            if (len(axis) - 1, iy) not in track:
                                track.append((len(axis) - 1, iy))
            track.sort()
            if track not in tracks and len(track) >= hit_threshold:
                tracks.append(track)
    all_straight_tracks.append(tracks)

import random
import math
import itertools

# alpha particles
# max_radius = 1
# all_alpha_tracks = []



# for max_radius in [2]:
    # for ix in range(len(axis)):
        # for iy in range(len(axis)):
            # x_ranges = [(ix - min_r, ix + max_r + 1)
                # for min_r, max_r in itertools.product(range(1, max_radius), range(max_radius))] 
            # y_ranges = [(iy - min_r, iy + max_r + 1)
                # for min_r, max_r in itertools.product(range(1, max_radius), range(max_radius))] 

            # for (x_range, y_range) in itertools.product(x_ranges, y_ranges):
                # track = []
                # for ixr in range(*x_range):
                    # if ixr < 0 or ixr >= len(axis):
                        # continue
                    # for iyr in range(*y_range):
                        # if iyr < 0 or iyr >= len(axis):
                            # continue
                        # # randomize hits inside the circle
                        # # removed from here, can be done just before training
                        # # rnd = random.random()
                        # # val = ((ixr - ix) ** 2 + (iyr - iy) ** 2) / (5. * max_radius)
                        # # if rnd > val:
                        # track.append((ixr, iyr))
                # track.sort()
                # if len(track) >= hit_threshold and track not in all_alpha_tracks:
                    # all_alpha_tracks.append(track)



# random noise
random_threshold = 0.02

def randomize(col, ntimes = 5):
    new_col = []
    for track in col:
        for time in range(ntimes):
            new_track = []
            for ix in range(len(axis)):
                for iy in range(len(axis)):
                    rnd = random.random()
                    if ((rnd >= random_threshold and (ix, iy) in track)
                            or (rnd < random_threshold and (ix, iy) not in track)):
                        new_track.append((ix, iy))
            if len(track) >= hit_threshold:
                new_col.append(new_track)
    return new_col


# alpha_tracks = randomize(all_alpha_tracks, 6)
straight_tracks = randomize(all_straight_tracks[0], 3)


random_tracks = []
while len(random_tracks) < len(straight_tracks):
# while 2 * len(random_tracks) < (len(alpha_tracks) + len(straight_tracks)):
    track = []
    for ix in range(len(axis)):
        for iy in range(len(axis)):
            rnd = random.random()
            if rnd < random_threshold:
                track.append((ix, iy))
    if len(track) >= hit_threshold:
        random_tracks.append(track)

d = []

 
# for name, collection in [((1, -1, -1), straight_tracks), ((-1, 1, -1), alpha_tracks), ((-1, -1, 1), random_tracks)]:
for name, collection in [((1,-1), straight_tracks), ((-1,1), random_tracks)]:
    for track in collection:
        track_to_add = []
        for ix in range(len(axis)):
            for iy in range(len(axis)):
                if (ix, iy) in track:
                    track_to_add.append(1)
                else:
                    track_to_add.append(0)
        d.append((track_to_add, name))

import json
with open("data.txt", "w+") as f:
    json.dump(d, f, indent=2)


# print(len(straight_tracks), len(random_tracks))
print(len(straight_tracks), len(random_tracks))

# extra code
# if len(d_slopes) > 1:
    # additional_1_2 = [track for track in all_tracks[0] if track not in all_tracks[1]]
    # additional_2_1 = [track for track in all_tracks[1] if track not in all_tracks[0]]

    # print len(additional_1_2), len(additional_2_1)
    
    
    # for track in additional_1_2:
        # for y in range(len(axis) - 1, -1, -1):
            # for x in range(len(axis)):
                # if (x, y) in track:
                    # print "1 ",
                # else:
                    # print "0 ",
            # print
        # a = raw_input()
        # if a == "":
            # continue

# from pprint import pprint
#pprint(tracks)


# for track in straight_tracks:
    # for y in range(len(axis) - 1, -1, -1):
        # for x in range(len(axis)):
            # if (x, y) in track:
                # print ("1 ", end="")
            # else:
                # print ("0 ", end="")
        # print()
    # a = input()
    # if a == "":
        # continue
