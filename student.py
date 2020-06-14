class Student:
    def __init__(self, name, age, program, grades, gpa):
        self.name = name
        self.age = age
        self.program = program
        #grades is a list
        self.grades = grades
        self.gpa = gpa

    def calculate_gpa(self, grade):
	    #add grade to list
        self.grades.append(grade)

        #update GPA
        self.gpa = sum(self.grades)/len(self.grades)

        print(self.grades)
        print(self.gpa)

    def print_fails (self):
        for grade in self.grades:
            if grade < 50:
                print(grade)