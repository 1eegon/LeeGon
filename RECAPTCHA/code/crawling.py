from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

driver = webdriver.Chrome()
# driver.get("https://www.google.co.kr/imghp?hl=ko")
# elem = driver.find_element_by_name("q")  #검색창찾기
# elem.send_keys("traffic light") #키보드 입력값 send
# elem.send_keys(Keys.RETURN) # Enter 입력
keyword=input("keyword 입력 : ")
path ="https://www.google.co.kr/search?as_st=y&tbm=isch&hl=ko&as_q="+keyword+"&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=itp:photo,iar:s,ift:jpg"
driver.get(path)



#스크롤 내리기
SCROLL_PAUSE_TIME = 2

# Get scroll height  - 자바스크립트 이용
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element_by_css_selector(".mye4qd").click()
        except:
            break;
    last_height = new_height

images= driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
count=1
savepath="C:/Users/eiden/PycharmProjects/pythonProject/images/"
for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl=driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img").get_attribute("src")
        urllib.request.urlretrieve(imgUrl, savepath+str(count)+".jpg")
        count+=1
    except:
        pass


driver.close()
