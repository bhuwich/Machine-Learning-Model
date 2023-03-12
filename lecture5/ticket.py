import pyautogui as pg
import time



time.sleep(5)
# click Origin
pg.moveTo(580, 918)
pg.click()
pg.moveTo(643, 655)
pg.click()
pg.write('Rangsit')
pg.press('down')
pg.press('enter')

# click Destination
pg.moveTo(762, 918)
pg.click()
pg.moveTo(833, 655)
pg.click()
pg.write('Chiang Mai')
pg.press('enter')

# click Travel Date
pg.moveTo(946, 918)
pg.click()
pg.moveTo(1043, 649)
pg.click()
pg.moveTo(932, 787) # Real travel date
# pg.moveTo(878, 787) # Fake travel date
pg.click()

# click Passenger
pg.moveTo(1335, 915)
pg.click()
pg.moveTo(1299, 651)
pg.click()

# click Search
pg.moveTo(1468, 918)
pg.click()

# wait for website to load
time.sleep(1)

# move page down
pg.moveTo(1911, 502)
pg.click()

# click Special CNR
pg.moveTo(1430, 387)
pg.click()

# wait for website to load
time.sleep(1)

# click first class
# pg.moveTo(1425, 478) # Real
pg.moveTo(1425, 708) # Fake
pg.click()

# wait for website to load
time.sleep(1)

pg.moveTo(773, 951)
pg.click()

pg.moveTo(954, 1035)
pg.click()

pg.moveTo(1912, 468)
pg.click()

pg.moveTo(495, 250)
pg.click()
pg.write('0982501793')

# wait for website to load
time.sleep(1)

pg.moveTo(775, 726)
pg.click()

pg.moveTo(955, 816)
pg.click()

pg.moveTo(516, 873)
pg.click()
pg.write('0867780405')

# wait for website to load
time.sleep(1)

pg.moveTo(1912, 654)
pg.click()

pg.moveTo(776, 503)
pg.click()

pg.moveTo(954, 590)
pg.click()

pg.moveTo(486, 649)
pg.click()
pg.write('0827025414')

pg.moveTo(831, 1033)
pg.click()

pg.moveTo(1910, 849)
pg.click()





