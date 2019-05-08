import os
import operator

import pandas as pd

delimiter = ';'

bdi_codes = {
    '02': 'ROUND LOT',
    '05': 'BMFBOVESPA REGULATIONS SANCTION',
    '06': 'STOCKS OF COS. UNDER REORGANIZATION',
    '07': 'EXTRAJUDICIAL RECOVERY',
    '08': 'JUDICIAL RECOVERY',
    '09': 'TEMPORARY ESPECIAL MANAGEMENT',
    '10': 'RIGHTS AND RECEIPTS',
    '11': 'INTERVENTION',
    '12': 'REAL ESTATE FUNDS',
    '14': 'INVESTMENT CERTIFICATES / DEBENTURES / PUBLIC DEBT SECURITIES',
    '18': 'BONDS',
    '22': 'BONUSES (PRIVATE)',
    '26': 'POLICIES / BONUSES / PUBLIC SECURITIES',
    '30': '',
    '32': 'EXERCISE OF INDEX CALL OPTIONS',
    '33': 'EXERCISE OF INDEX PUT OPTIONS',
    '38': 'EXERCISE OF CALL OPTIONS',
    '42': 'EXERCISE OF PUT OPTIONS',
    '46': 'AUCTION OF NON-QUOTED SECURITIES',
    '48': 'PRIVATIZATION AUCTION',
    '49': 'AUCTION OF ECONOMICAL RECOVERY FUND OF ESPIRITO SANTO STATE',
    '50': 'AUCTION',
    '51': 'FINOR AUCTION',
    '52': 'FINAM AUCTION',
    '53': 'FISET AUCTION',
    '54': 'AUCTION OF SHARES IN ARREARS',
    '56': 'SALES BY COURT ORDER',
    '58': 'OTHERS',
    '60': 'SHARE SWAP',
    '61': 'GOAL',
    '62': 'TERM',
    '66': 'DEBENTURES WITH MATURITY DATES OF UP TO 3 YEARS',
    '68': 'DEBENTURES WITH MATURITY DATES GREATER THAN 3 YEARS',
    '70': 'FORWARD WITH CONTINUOUS MOVEMENT',
    '71': 'FORWARD WITH GAIN RETENTION',
    '74': 'INDEX CALL OPTIONS',
    '75': 'INDEX PUT OPTIONS',
    '78': 'CALL OPTIONS',
    '82': 'PUT OPTIONS',
    '83': 'DEBENTURES AND PROMISSORY NOTES BOVESPAFIX',
    '84': 'DEBENTURES AND PROMISSORY NOTES SOMAFIX',
    '90': 'REGISTERED TERM VIEW',
    '96': 'FACTIONARY',
    '99': 'GRAND TOTAL'
}

types_of_market = {
    '010': 'CASH',
    '012': 'EXERCISE OF CALL OPTIONS',
    '013': 'EXERCISE OF PUT OPTIONS',
    '017': 'AUCTION',
    '020': 'FACTIONARY',
    '030': 'TERM',
    '050': 'FORWARD WITH GAIN RETENTION',
    '060': 'FORWARD WITH CONTINUOUS MOVEMENT',
    '070': 'CALL OPTIONS',
    '080': 'PUT OPTIONS'
}


stats = {}


def count_stat(stat_name, key):
    if stat_name not in stats:
        stats[stat_name] = {}

    if key in stats[stat_name]:
        stats[stat_name][key] += 1
    else:
        stats[stat_name][key] = 1


def print_stat(stat_name):
    if stat_name in stats:
        stat = dict(reversed(sorted(stats[stat_name].items(), key=operator.itemgetter(1))))
        print('{}: {}'.format(stat_name, str(stat)))
    else:
        print('No stats for \'{}\''.format(stat_name))


def print_all_stats():
    for stat_name in stats.keys():
        print_stat(stat_name)


def paper_specification(spec):
    if spec[0:3] == 'BNS':
        return 'SUBSCRIPTION BONUS'
    elif spec[0:3] == 'CDA':
        return 'COMMON SHARE DEPOSIT CERTIFICATE'
    elif spec[0:2] == 'CI':
        return 'INVESTMENT FUND'
    elif spec[0:3] == 'DIR':
        return 'SUBSCRIPTION RIGHT'
    elif spec[0:3] == 'LFT':
        return 'FINANCIAL TREASURY BILL'
    elif spec[0:2] == 'ON':
        return 'COMMON SHARE'
    elif spec[0:2] == 'OR':
        return 'REDEEMABLE SHARE'
    elif spec[0:2] == 'PN':
        return 'PREFERRED SHARE'
    elif spec[0:2] == 'PR':
        return 'PREFERRED REDEEMABLE SHARE'
    elif spec[0:3] == '':
        return ''
    elif spec[0:3] == '':
        return ''
    else:
        return spec.strip()


def format_number(numstr, num_format='11.2'):
    dot_index = num_format.find('.')
    before_dot = int(num_format[0:dot_index])
    after_dot = int(num_format[dot_index+1:])
    if len(numstr) != before_dot + after_dot:
        raise Exception('Number string has to have length of {} + {}!'.format(before_dot, after_dot))
    return '{}.{}'.format(int(numstr[0:before_dot]), int(numstr[before_dot:]))


def transform_line(line, shares_of_interest):
    if line[0:2] == '99':
        return ''

    bdi = bdi_codes[line[10:12]]
    tmarket = types_of_market[line[24:27]]
    pspec = paper_specification(line[39:49])
    prazot = line[49:52].strip()
    currency = line[52:56].strip()
    company_name = line[27:39].strip()
    neg_code = line[12:24].strip()
    quotation_factor = str(int(line[210:217]))

    if neg_code not in shares_of_interest:
        return ''

    result = [
        '{}-{}-{}'.format(line[2:6], line[6:8], line[8:10]),  # Date
        # bdi,  # bdi
        neg_code,  # paper negotiation code
        company_name,  # company name
        # tmarket,  # type of market
        # pspec,  # paper specification
        # prazot,  # forward market term in days
        # currency,  # currency
        format_number(line[ 56: 69]),  # open
        format_number(line[ 69: 82]),  # high
        format_number(line[ 82: 95]),  # low
        format_number(line[ 95:108]),  # average
        format_number(line[108:121]),  # close
        str(int(line[147:152])),  # number of trades
        str(int(line[152:170])),  # number of titles traded
        format_number(line[170:188], '16.2'),  # trading volume
        quotation_factor,  # paper quotation factor
        line[230:242]  # ISIN
    ]

    count_stat('bdi', bdi)
    count_stat('types of market', tmarket)
    count_stat('paper spec', pspec)
    count_stat('prazot', prazot)
    count_stat('currency', currency)
    count_stat('neg_code', neg_code)
    count_stat('quotation_factor', quotation_factor)

    return '{}\n'.format(delimiter.join(result))


def convert_to_csv(shares_of_interest: list, orig_data_path: str, transformed_data_path: str, override: bool = False) -> None:
    """
    Prepare data if that is not done yet.
    :param orig_data_path:
    :param shares_of_interest:
    :param transformed_data_path:
    :return: Nothing
    """
    _, _, files = next(os.walk(transformed_data_path))
    if len(files) == 0 or override == True:
        print("\nReading data from\n\t{}\nand storing transformed data in\n\t{}.".format(orig_data_path,
                                                                                         transformed_data_path))
        root, _, files = next(os.walk(orig_data_path))
        for file in files:
            print("Processing {}...".format(file))
            with open(os.path.join(root, file), 'r', encoding='latin-1') as in_f:
                year = in_f.readline().strip()[11:15]
                with open(os.path.join(transformed_data_path, '{}.txt'.format(year)), 'w', encoding='utf-8') as out_f:
                    out_f.write(delimiter.join([
                        'date',
                        # 'bdi',
                        'symbol',  # 'paper negotiation code'
                        'company name',
                        # 'type of market',
                        # 'paper spec',
                        # 'prazot',
                        # 'currency',
                        'open',
                        'high',
                        'low',
                        'avg',
                        'close',
                        '#trades',
                        '#titles traded',
                        'volume',
                        'quotation factor',
                        'ISIN']))
                    out_f.write('\n')
                    for line in in_f:
                        out_f.write(transform_line(line, shares_of_interest))


def load_data(transformed_data_path: str) -> dict:
    """
    Read stock data and concat to one big DataFrame. Then group stock data by stock name (symbol).
    :param transformed_data_path: Path to data in csv format as created by bovespa_data.transform_data()
    :return: dict: Mapping stock name symbol to DataFrame with index on date.
    """
    df = pd.DataFrame()
    _, _, files = next(os.walk(transformed_data_path))
    for file in files:
        path = os.path.join(transformed_data_path, file)
        df_tmp = pd.read_csv(path, delimiter=';', index_col=False)
        df_tmp['date'] = pd.to_datetime(df_tmp['date'], format='%Y-%m-%d')
        df_tmp = df_tmp.sort_values(['symbol', 'date'])

        df = pd.concat([df, df_tmp], axis=0)
    df.set_index(['symbol', 'date'], inplace=True)
    df.sort_index(inplace=True)
    dfs = {}
    for stock in df.index.levels[0]:
        dfs[stock] = df.loc[stock]
    return dfs


if __name__ == '__main__':

    shares_of_interest = [
        'AMBV4',
        'ARCZ6',
        'BBAS3',
        'BBDC4',
        'CMIG4',
        'CRUZ3',
        'CSNA3',
        'ELET6',
        'ITAU4',
        'ITSA4',
        'NETC4',
        'PETR4',
        'TNLP4',
        'USIM5',
        'VALE5'
    ]

    orig_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/bovespa/orig_data')
    transformed_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/bovespa/data')

    convert_to_csv(shares_of_interest, orig_data_path, transformed_data_path)

    print_all_stats()
