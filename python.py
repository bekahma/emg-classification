import student as s

def two_ints(N,M):
	list = []
	for i in range(N):
		if i%M == 0:
			list.append(i)
	print(list)


#defining the function
def one_int(N):
	dict = {}
	squares = []
	powers = []
	for i in range(N**2+1):
		if i**(1/2) <= N and (i**(1/2))%1==0:
			squares.append(i)
		if 2**i <=N:
			powers.append(i)
	dict["squares"] = squares
	dict["powers"] = powers
	print(dict)

if __name__ == '__main__':
	# question 7a
	# two_ints(12, 2)
	# # calling the function
	# one_int(4)
	# one_int(3)

	#testing Student class
	student1 = s.Student("Lixin", 19, "BME", [100, 30, 49, 101], 70)
	student1.calculate_gpa(90)
	student1.print_fails()