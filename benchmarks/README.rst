v1 

k_nearest:
 - computes the distance to all other states
 - sorts the whole state list by this distance
 - truncates the state list to k

v2

use itertools::k_smallest_by

v3

actually use dubins to compute the costs...
this is cause k_nearest to be called way more
can't reuse k_nearest searches between finding the best parent and updating weights of the children
this is because dubins distance is different depending on whether you are doing a -> b or b -> a
this is different than the euclidean distance

v4

use euclidean distance filter for k_nearest with 2*k bound
