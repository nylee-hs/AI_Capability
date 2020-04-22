import schedule
import time

def job():
    print('working...')

def main():
    schedule.every().day.at("09:21").do(job)
    schedule.every().day.at("09:22").do(job)
    # schedule.every(10).seconds.do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()