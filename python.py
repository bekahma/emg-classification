def two_ints(N,M):
	list = []
	for i in range(N):
		if i%M == 0:
			list.append(i)
	print(list)
#calling the function
two_ints(12,2)

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

#calling the function
one_int(4)
one_int(3)
