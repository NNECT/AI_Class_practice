from flask import Flask, render_template
import random


# 127.0.0.1 -> local loopback
# 192.0.0.1 -> private ip
# 120.0.0.1 -> public ip


app = Flask('my first web server')


@app.route('/')
def index():
    return 'Hello, flask!'


def make_randnums(size, limit):
    return [random.randrange(limit) for _ in range(size)]


@app.route('/randnums')
def show_randnums():
    return str(make_randnums(10, 100))


@app.route('/lotto')
def lotto_numbers():
    # nums = []
    # while len(nums) < 6:
    #     num = random.randrange(45) + 1
    #     if num not in nums:
    #         nums.append(num)
    # nums.sort()
    # return '추첨된 로또 번호: ' + ' '.join([str(num) for num in nums])
    return '추첨된 로또 번호: ' + ' '.join([str(n) for n in (sorted(random.sample(list(range(1, 46)), 6)))])


@app.route('/lotto645')
def lotto645():
    nums = sorted(random.sample(list(range(1, 46)), 6))
    return render_template('lotto645.html', nums=nums)


if __name__ == '__main__':
    app.run(debug=True)

# static, templates
