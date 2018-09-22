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

	me = {
		"thai": 3,
		"japanese": 2,
		"american": 1
	}

	users = [user_1, user_2, user_3, user_4]

	print(rec.recommend_me(me, users, n=1, intersection=True))
	print(rec.recommend_me(me, users, n=1, intersection=False))

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

def test_cosine_similarity():

	rec = recommender.Recommender()

	a = np.array([1, 0, 0])
	b = np.array([0, 1, 0])
	similarity = rec.cosine_similarity(a, b)
	assert similarity == 0

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


if __name__ == '__main__':

	# unit tests
	# test_cosine_similarity()
	# test_dicts_to_vectors()
	# test_find_similarity()
	test_all()
