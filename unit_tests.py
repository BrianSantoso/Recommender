import recommender
import numpy as np

def test_all():

	rec = recommender.Recommender()

	user_1 = {
		"vegan": 1,
		"japanese": 4
	}

	user_2 = {
		"vegan": 5,
		"halal": 1
	}

	user_3 = {
		"american": 4,
		"mexican": 3,
		"japanese": 2
	}

	user_4 = {
		"thai": 4,
		"mexican": 1,
		"japanese": 4
	}

	user_5 = {
	}

	user_6 = {
		"thai": 3,
		"japanese": 2,
		"american": 1
	}

	me = {
		"thai": 3,
		"japanese": 2,
		"american": 1
	}

	users = [user_1, user_2, user_3, user_4, user_5, user_6]


	print(rec.recommend_me(me, users, 6, intersection=True, with_keys=False))
	print(rec.recommend_me(me, users, 6, intersection=False, with_keys=False))

	# most similar user with strongest shared interests
	index = rec.recommend_me(me, users, 1, intersection=True, with_keys=False)[0]
	print(index, rec.find_most_shared_interests(me, users[index], 3))
	# print("index 0:", rec.find_similarity(me, user_1))
	# print("index 3:", rec.find_similarity(me, user_4))

def test_find_most_shared_interests():
	rec = recommender.Recommender()
	me = {
		"thai": 3,
		"japanese": 2,
		"american": 1
	}

	user_1 = {
		"vegan": 1,
		"japanese": 4
	}

	print(rec.find_most_shared_interests(me, user_1, 1))

	me = {
		"thai": 3,
		"japanese": 2,
		"american": 1
	}
	user_5 = {
	}
	print(rec.find_most_shared_interests(me, user_5, 3))

	user_3 = {
		"american": 4,
		"mexican": 3,
		"japanese": 2
	}
	x = {
		"thai": 4,
		"mexican": 1,
		"japanese": 4,
		"american": 4
	}
	print(rec.find_most_shared_interests(user_3, x, 3))

def test_find_similarity():

	rec = recommender.Recommender()
	john = {
		"vegan": 1
	}
	joe = {
		"vegan": 5,
		"halal": 1
	}
	print(rec.find_similarity(john, joe))

	john = {
		"vegan": 1
	}
	joe = {
	}
	print("empty case:", rec.find_similarity(john, joe))

def test_cosine_similarity():

	rec = recommender.Recommender()

	a = np.array([1, 0, 0])
	b = np.array([0, 1, 0])
	similarity = rec.cosine_similarity(a, b)
	assert similarity == 0

	a = np.array([1, 0, 0])
	b = np.array([])
	similarity = rec.cosine_similarity(a, b)
	print("cos sim empty case:",similarity)

def test_dicts_to_vectors():

	john = {
		"vegan": 1
	}
	joe = {
		"vegan": 1,
		"halal": 1
	}
	rec = recommender.Recommender()
	print(rec.dicts_to_vectors(john, joe))


	john = {
		"vegan": 1
	}
	joe = {
	}
	rec = recommender.Recommender()
	print("dicts to vecs empty case:", rec.dicts_to_vectors(john, joe))


if __name__ == '__main__':

	# unit tests
	# test_cosine_similarity()
	# test_dicts_to_vectors()
	# test_find_similarity()
	# test_find_most_shared_interests()
	test_all()
