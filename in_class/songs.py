import os
import pickle
import re
import requests


def download_songs_data(code, page):
    filename = f'singer_songs/{code}_{page}.pickle'
    payload = {
        'S_PAGENUMBER': page,
        'S_MB_CD': code,
    }
    url = 'https://www.komca.or.kr/srch2/srch_01_popup_mem_right.jsp'
    if not os.path.exists(filename):
        with open(filename, 'wb') as f:
            res = requests.post(url, data=payload)
            pickle.dump(res.text, f)
    # else:
    #     print(f'{filename}: already downloaded')


def get_songs_current_page(code, page):
    download_songs_data(code, page)
    filename = f'singer_songs/{code}_{page}.pickle'
    with open(filename, 'rb') as f:
        text = pickle.load(f)

    # 테이블 부분 추출
    table = re.findall(r'<tbody>(.+?)</tbody>', text, re.DOTALL)[1]
    data = re.findall(r'<tr>(.+?)</tr>', table, re.DOTALL)
    # 저작물 목록 추출
    rows = [re.findall(r'<td>(.*)</td>', t) for t in data]
    # 각 목록에서 요소 추출
    rows = [[[re.sub('<.+>', '', elm).strip() for elm in column.strip().split('<br/>')] for column in row] for row in rows]

    return rows


def get_songs(code):
    data = []
    n = 1
    while True:
        page_data = get_songs_current_page(code, n)
        if len(page_data) == 0:
            break
        data += page_data
        n += 1
    return data


def get_songs_by_name(name):
    code = fine_code(name)
    return get_songs(code)


def fine_code(name):
    payload = {
        'SYSID': 'PATHFINDER',
        'MENUID': 1000005022001,
        'EVENTID': 'srch_01_popup_mem',
        'S_MB_CD': '',
        'S_MB_CD_POPUP': 1,
        'S_HANMB_NM': name,
        'hanmb_nm': '',
        'form_val': '',
        'form_val2': '',
        'input_idx': '',
        'input_name': 'S_RIGHTPRES_CD',
        'input_name2': 'S_RIGHTPRES_NM',
        'pub_val': 0,
        'search_code': 1,
        'search_keyword': name
    }
    url = 'https://www.komca.or.kr/CTLJSP'
    res = requests.post(url, data=payload)

    table = re.findall(r'<tbody>(.+?)</tbody>', res.text, re.DOTALL)[0]
    tr = re.findall(r'<tr>(.+?)</tr>', table, re.DOTALL)[0]
    code = re.findall(r'>(.+?)<', tr)[1]

    return code


if __name__ == '__main__':
    # get 방식:   쉽다       길이 제한이 있다      암호화가 불가능하다
    # post 방식:  불편하다    길이 제한이 없다      암호화가 가능하다      폼데이터 처리가 가능하다

    g = get_songs_by_name('지드래곤')

    # for row in g:    # 각 저작물 목록
    #     for column in row:  # 저작물 목록 정보 종류
    #         print(*column, sep=', ')
    #     print()
    print(len(g))


# applekoong@naver.com 김정훈