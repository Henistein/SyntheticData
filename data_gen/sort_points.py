import math

class Point:
  def __init__(self, pt):
    self.x = pt[0]
    self.y = pt[1]

  def __getitem__(self, items):
    if items == 0:
      return self.x
    elif items == 1:
      return self.y
    else:
      raise IndexError
  def __setitem__(self, key, value):
    if key == 0:
        self.x = value
    elif key == 1:
        self.y = value
    else:
        raise IndexError


def sort_points_cw(points):
  pt = [0, 0]

  for point in points:
    pt[0] = pt[0] + point[0]
    pt[1] = pt[1] + point[1]

  pt[0] = pt[0]/len(points)
  pt[1] = pt[1]/len(points)

  for point in points:
    point[0] = point[0] - pt[0]
    point[1] = point[1] - pt[1]

  points = sorted(points, key=compare_points)

def compare_points(pts):
  print(pts)
  pt1, pt2 = pts
  angle1 = get_angle([0,0], pt1)
  angle2 = get_angle([0,0], pt2)

  if angle1 < angle2:
    return (pt1, pt2)
  
  d1 = get_distance([0,0], pt1)
  d2 = get_distance([0,0], pt2)

  if (angle1 == angle2) and (d1 < d2):
    return (pt1, pt2)

  return (pt1, pt2)

def get_angle(pt_center, point):
  x = point[0] - pt_center[0]
  y = point[1] - pt_center[1]
  angle = math.atan2(y, x)
  if angle <= 0:
    angle = 2 * math.pi + angle
  return angle

def get_distance(pt1, pt2):
  x = pt1[0] - pt2[0]
  y = pt1[1] - pt2[1]
  return math.sqrt(x*x + y*y)


pts = [[4, 3], [4, 3], [4, 336], [3, 336]]
pts = list(map(Point, pts))
print(sort_points_cw(pts))
