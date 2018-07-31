

store_number = []
end_key = 1
while end_key != 0:
    end_key= input('Write a number ')
    try:
        end_key = int(end_key)
        if end_key != 0:
            store_number.append(end_key)
    except ValueError:
        print('input is not a number')
        pass


for number in store_number:
    print(number)
minSum = min(store_number)
maxSum = max(store_number)
print('min element: ',minSum, '\n','max element: ' , maxSum)





