farenheit = input('Skriv inn temperatur i farenheit: ')
farenheit = int(farenheit)
print(str(farenheit)+'F')

celcius = (farenheit-32)*5/9

print('Konvertert temperatur: ', str(round(celcius, 2))+'C')