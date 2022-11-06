import math
import itertools

def find_centroid(pts):
  x = 0
  y = 0
  for p in pts:
    x += p[0]
    y += p[1]

  center = [0,0]
  center[0] = x / len(pts)
  center[1] = y / len(pts)

  return center

def comp(pair):
  center = [3.75, 169.5]
  if pair[1] is None:
    return (math.degrees(math.atan2(pair[0][0]-center[0],pair[0][1]-center[1]))+360)%360

  return ((math.degrees(math.atan2(pair[0][0]-center[0],pair[0][1]-center[1]))+360)%360 - \
          (math.degrees(math.atan2(pair[1][0]-center[0],pair[1][1]-center[1]))+360)%360)

def sort_vertices(pts):
  center = find_centroid(pts)
  return [outpair[0] for outpair in sorted(itertools.zip_longest(pts, pts[1:]), key=comp)]



pts = [[4, 3], [4, 3], [4, 336], [3, 336]]
print(sort_vertices(pts))
