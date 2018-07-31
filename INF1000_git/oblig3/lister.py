liste = [1, 4, 8]
print(liste[0], liste[2])

navneliste = []

for i in range(4):
    navn = input('Skriv inn navn: ')
    navneliste.append(navn)

print(navneliste)

sjekkNavn = input('sjekk dette navnet: ')

if sjekkNavn in navneliste:
    print('du husket meg')
else:
    print('glemte du meg?')

kombinertListe = liste
print(kombinertListe)
kombinertListe = kombinertListe+ navneliste
print(kombinertListe)
del kombinertListe[-2:]
print('fjerner de to siste elementene:', kombinertListe)