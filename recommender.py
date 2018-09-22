import numpy as np
from numpy import linalg as LA

class Recommender():

	def __init__(self):

		return

	def recommend_me(self, me, all_other_users, n, intersection=True, with_keys=False):
		# returns indices of most similar users with
		# the intersection or union of their shared interests
		# Note: make sure me is not in all_other_users

		similarity_values_with_keys = [self.find_similarity(me, user, intersection) for user in all_other_users]
		similarity_values = np.asarray([similarity_values_with_keys[i][0] for i in range(len(similarity_values_with_keys))])

		top_n = similarity_values.argsort()[-n:][::-1]

		if with_keys:
			output = []
			for index in top_n:
				output.append((index, similarity_values_with_keys[index][1]))

			return output
		else:
			return top_n

		

	# def find_shared_interests(self, user1, user2, n=1):
	# 	# finds top n shared interests between two users
	# 	a, b, keys = dicts_to_vectors(user1, user2, intersection=True)
	# 	top_n = similarity_values.argsort()[-n:][::-1]

	def find_similarity(self, user1, user2, intersection=True):
		# accepts 2 dicts
		#
		# if intersection is True, users will be compared only based
		# on things they have BOTH viewed (intersection)
		# else, they will be compared on everything they viewed (union)

		a, b, keys = self.dicts_to_vectors(user1, user2, intersection)

		similarity = self.cosine_similarity(a, b)

		return similarity, keys


	def cosine_similarity(self, a, b):
		# accepts 2 vectors
		# returns similarity metric between 2 vectors in the range [1, -1]

		if a.size == 0 or b.size == 0:
			# if vector(s) is empty, then it will be nan, so return 0 (neutral similarity)
			return 0

		dot = np.dot(a, b)
		cos_theta = dot / (LA.norm(a) * LA.norm(b))

		return cos_theta

	def dicts_to_vectors(self, a, b, intersection=True):

		# finds either the intersection or union of 2 vectors,
		# then then returns array of values for their corresponding values,
		# padding them with 0's if using union and the values don't exist

		a_vector = []
		b_vector = []
		keys = []

		if intersection:

			keys_a = set(a.keys())
			keys_b = set(b.keys())
			inter = keys_a & keys_b
			keys = inter

			for key in inter:

				a_vector.append(a[key])
				b_vector.append(b[key])

		else:
			# list of all keys
			union = dict(b, **a).keys()
			keys = union
			
			for key in union:
				a_vector.append(a[key]) if key in a else a_vector.append(0)
				b_vector.append(b[key]) if key in b else b_vector.append(0)


		return np.asarray(a_vector), np.asarray(b_vector), keys



