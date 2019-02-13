import argparse
import csv
import glob
import os
from collections import Counter
from yattag import Doc

def get_csvfiles_from_dir(report_dir):
    os.chdir(report_dir)
    for f in glob.glob("*.csv"):
        yield f

def get_predictions_from_csvfiles(files):
    all_results = {}
    for file_name in files:
        with open(file_name) as csv_file:
            file_results = []
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) > 1:
                    file_results.append([row[0][2:]] + row[-6:])
            all_results[file_name] = file_results
    return all_results

def get_tactic_distribution(predictions):
    tactic_distribution = {}
    all_tactic_distribution = {}
    for file_name in predictions:
        file_predictions = predictions[file_name]
        file_tactic_distribution = {}
        for prediction in file_predictions:
            correct_tactic = prediction[0]
            if correct_tactic not in file_tactic_distribution:
                file_tactic_distribution[correct_tactic] = Counter()
            if correct_tactic not in all_tactic_distribution:
                all_tactic_distribution[correct_tactic] = Counter()
            # file_tactic_distribution[correct_tactic]["found"] += 1
            for i in range(3):
                curr_tactic = prediction[1+i*2]
                tactic_status = prediction[2+i*2]
                if curr_tactic not in file_tactic_distribution:
                    file_tactic_distribution[curr_tactic] = Counter()
                if curr_tactic not in all_tactic_distribution:
                    all_tactic_distribution[curr_tactic] = Counter()
                if i == 0:
                    file_tactic_distribution[correct_tactic]["human_top_1_total"] += 1
                    file_tactic_distribution[curr_tactic]["bot_top_1_total"] += 1
                    all_tactic_distribution[correct_tactic]["human_top_1_total"] += 1
                    all_tactic_distribution[curr_tactic]["bot_top_1_total"] += 1
                    if tactic_status == "goodcommand":
                        file_tactic_distribution[correct_tactic]["human_top_1_anygood"] += 1
                        file_tactic_distribution[correct_tactic]["human_top_1_good"] += 1
                        file_tactic_distribution[curr_tactic]["bot_top_1_good"] += 1
                        all_tactic_distribution[correct_tactic]["human_top_1_anygood"] += 1
                        all_tactic_distribution[correct_tactic]["human_top_1_good"] += 1
                        all_tactic_distribution[curr_tactic]["bot_top_1_good"] += 1
                    elif tactic_status == "mostlygoodcommand":
                        file_tactic_distribution[correct_tactic]["human_top_1_anygood"] += 1
                        file_tactic_distribution[correct_tactic]["human_top_1_mostlygood"] += 1
                        file_tactic_distribution[curr_tactic]["bot_top_1_mostlygood"] += 1
                        all_tactic_distribution[correct_tactic]["human_top_1_anygood"] += 1
                        all_tactic_distribution[correct_tactic]["human_top_1_mostlygood"] += 1
                        all_tactic_distribution[curr_tactic]["bot_top_1_mostlygood"] += 1
                    elif tactic_status == "badcommand":
                        file_tactic_distribution[correct_tactic]["human_top_1_anybad"] += 1
                        file_tactic_distribution[correct_tactic]["human_top_1_bad"] += 1
                        file_tactic_distribution[curr_tactic]["bot_top_1_bad"] += 1
                        all_tactic_distribution[correct_tactic]["human_top_1_anybad"] += 1
                        all_tactic_distribution[correct_tactic]["human_top_1_bad"] += 1
                        all_tactic_distribution[curr_tactic]["bot_top_1_bad"] += 1
                    elif tactic_status == "failedcommand":
                        file_tactic_distribution[correct_tactic]["human_top_1_anybad"] += 1
                        file_tactic_distribution[correct_tactic]["human_top_1_failed"] += 1
                        file_tactic_distribution[curr_tactic]["bot_top_1_failed"] += 1
                        all_tactic_distribution[correct_tactic]["human_top_1_anybad"] += 1
                        all_tactic_distribution[correct_tactic]["human_top_1_failed"] += 1
                        all_tactic_distribution[curr_tactic]["bot_top_1_failed"] += 1
                    else:
                        print(tactic_status)
                        exit()
                file_tactic_distribution[correct_tactic]["human_top_3_total"] += 1
                file_tactic_distribution[curr_tactic]["bot_top_3_total"] += 1
                all_tactic_distribution[correct_tactic]["human_top_3_total"] += 1
                all_tactic_distribution[curr_tactic]["bot_top_3_total"] += 1
                if tactic_status == "goodcommand":
                    file_tactic_distribution[correct_tactic]["human_top_3_good"] += 1
                    file_tactic_distribution[curr_tactic]["bot_top_3_good"] += 1
                    all_tactic_distribution[correct_tactic]["human_top_3_good"] += 1
                    all_tactic_distribution[curr_tactic]["bot_top_3_good"] += 1
                elif tactic_status == "mostlygoodcommand":
                    file_tactic_distribution[correct_tactic]["human_top_3_mostlygood"] += 1
                    file_tactic_distribution[curr_tactic]["bot_top_3_mostlygood"] += 1
                    all_tactic_distribution[correct_tactic]["human_top_3_mostlygood"] += 1
                    all_tactic_distribution[curr_tactic]["bot_top_3_mostlygood"] += 1
                elif tactic_status == "badcommand":
                    file_tactic_distribution[correct_tactic]["human_top_3_bad"] += 1
                    file_tactic_distribution[curr_tactic]["bot_top_3_bad"] += 1
                    all_tactic_distribution[correct_tactic]["human_top_3_bad"] += 1
                    all_tactic_distribution[curr_tactic]["bot_top_3_bad"] += 1
                elif tactic_status == "failedcommand":
                    file_tactic_distribution[correct_tactic]["human_top_3_failed"] += 1
                    file_tactic_distribution[curr_tactic]["bot_top_3_failed"] += 1
                    all_tactic_distribution[correct_tactic]["human_top_3_failed"] += 1
                    all_tactic_distribution[curr_tactic]["bot_top_3_failed"] += 1
                else:
                    print(tactic_status)
                    exit()
        tactic_distribution[file_name] = file_tactic_distribution
        # import pprint
        # pprint.pprint(file_name)
        # pprint.pprint(file_tactic_distribution)
        # pprint.pprint("===================")
    tactic_distribution["all"] = all_tactic_distribution
    return tactic_distribution

def report_distributions(tactic_distributions):
    doc, tag, text, line = Doc().ttl()
    with tag('html'):
        with tag('head'):
            doc.stag('link', href="report.css", rel='stylesheet')
            with tag('script', type='text/javascript', src="report.js"):
                pass
            with tag('title'):
                text("Proverbot Analysis")
        with tag('body'):
            with tag('table'):
                with tag('tr', klass="header"):
                    line('th', 'Filename')
                    line('th', 'Tactic')
                    line('th', 'H-1 Total')
                    line('th', 'H-1 AnyGood')
                    line('th', 'H-1 Good')
                    line('th', 'H-1 Mostly Good')
                    line('th', 'H-1 AnyBad')
                    line('th', 'H-1 Bad')
                    line('th', 'H-1 Failed')
                    line('th', 'H-3 Total')
                    line('th', 'H-3 Good')
                    line('th', 'H-3 Mostly Good')
                    line('th', 'H-3 Bad')
                    line('th', 'H-3 Failed')
                    line('th', 'B-1 Total')
                    line('th', 'B-1 Good')
                    line('th', 'B-1 Mostly Good')
                    line('th', 'B-1 Bad')
                    line('th', 'B-1 Failed')
                    line('th', 'B-3 Total')
                    line('th', 'B-3 Good')
                    line('th', 'B-3 Mostly Good')
                    line('th', 'B-3 Bad')
                    line('th', 'B-3 Failed')
                    # line('th', 'Human Top-1 Total')
                    # line('th', 'Human Top-1 Good')
                    # line('th', 'Human Top-1 Mostly Good')
                    # line('th', 'Human Top-1 Bad')
                    # line('th', 'Human Top-1 Failed')
                    # line('th', 'Human Top-3 Total')
                    # line('th', 'Human Top-3 Good')
                    # line('th', 'Human Top-3 Mostly Good')
                    # line('th', 'Human Top-3 Bad')
                    # line('th', 'Human Top-3 Failed')
                    # line('th', 'Bot Top-1 Total')
                    # line('th', 'Bot Top-1 Good')
                    # line('th', 'Bot Top-1 Mostly Good')
                    # line('th', 'Bot Top-1 Bad')
                    # line('th', 'Bot Top-1 Failed')
                    # line('th', 'Bot Top-3 Total')
                    # line('th', 'Bot Top-3 Good')
                    # line('th', 'Bot Top-3 Mostly Good')
                    # line('th', 'Bot Top-3 Bad')
                    # line('th', 'Bot Top-3 Failed')
                rows = [(filename, tactic, tactic_distributions[filename][tactic]) for filename in tactic_distributions for tactic in tactic_distributions[filename]]
                # while rows.qsize() > 0:
                    # sorted_rows.append(rows.get())
                sorted_rows = sorted(rows, key=lambda tup: (tup[0], tup[1]))

                for file_name, tactic, dist in sorted_rows:
                    # if fresult.num_tactics == 0:
                    #     continue
                    with tag('tr'):
                        line('td', file_name)
                        line('td', tactic)
                        line('td', dist["human_top_1_total"])
                        line('td', dist["human_top_1_anygood"])
                        line('td', dist["human_top_1_good"])
                        line('td', dist["human_top_1_mostlygood"])
                        line('td', dist["human_top_1_anybad"])
                        line('td', dist["human_top_1_bad"])
                        line('td', dist["human_top_1_failed"])
                        line('td', dist["human_top_3_total"])
                        line('td', dist["human_top_3_good"])
                        line('td', dist["human_top_3_mostlygood"])
                        line('td', dist["human_top_3_bad"])
                        line('td', dist["human_top_3_failed"])
                        line('td', dist["bot_top_1_total"])
                        line('td', dist["bot_top_1_good"])
                        line('td', dist["bot_top_1_mostlygood"])
                        line('td', dist["bot_top_1_bad"])
                        line('td', dist["bot_top_1_failed"])
                        line('td', dist["bot_top_3_total"])
                        line('td', dist["bot_top_3_good"])
                        line('td', dist["bot_top_3_mostlygood"])
                        line('td', dist["bot_top_3_bad"])
                        line('td', dist["bot_top_3_failed"])
                        # with tag('td'):
                        #     with tag('a', href=fresult.details_filename() + ".html"):
                        #         text("Details")
                # with tag('tr'):
                #     line('td', "Total");
                #     line('td', str(self.num_tactics))
                #     line('td', str(self.num_searched))
                #     line('td', stringified_percent(self.num_searched,
                #                                     self.num_tactics))
                #     line('td', stringified_percent(self.num_correct,
                #                                     self.num_tactics))
                #     line('td', stringified_percent(self.num_topN,
                #                                     self.num_tactics))
                #     line('td', stringified_percent(self.num_partial,
                #                                     self.num_tactics))
                #     line('td', stringified_percent(self.num_topNPartial,
                #                                     self.num_tactics))
                #     line('td', "{:10.2f}".format(self.total_loss / self.num_tactics))
    return doc

def main() -> None:
    parser = argparse.ArgumentParser(description="Report perfomance by tactic")
    parser.add_argument("-d", "--report-dir", dest="report_dir", type=str)
    args = parser.parse_args()

    files = get_csvfiles_from_dir(args.report_dir)
    predictions = get_predictions_from_csvfiles(files)
    tactic_distributions = get_tactic_distribution(predictions)
    doc = report_distributions(tactic_distributions)
    # with open("{}/analysis.html".format(args.report_dir), "w") as fout:
    with open("analysis.html", "w") as fout:
        fout.write(doc.getvalue())

if __name__ == "__main__":
    main()