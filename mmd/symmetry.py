N = 2

count = 0
ij = 0
for i in range(N):
    for j in range(N):
        ij += 1
        kl = 0
        for k in range(N):
            for l in range(N):
                kl += 1
                if kl <= ij:
                    print (i,j,k,l)
                    count += 1
                    if (i,j,k,l) != (k,l,i,j):
                        print (k,l,i,j)           

count2 = 0
print "\n"
for i in range(N):
    for j in range(N):
        for k in range(N):
            for l in range(N):
                print (i,j,k,l) 
                count2 +=1

print count, count2
