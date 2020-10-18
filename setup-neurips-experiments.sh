
# WS experiments (won't get the same instances that we had pre NeurIPS)
#mkdir experiments/suite-WATTS
#python3 experiments/generate_suite.py -o experiments/suite-watts/ -g WATTS --min-vars 3 --max-vars 15 --rep 5

# LADDER experiments
mkdir experiments/suite-LADDER
python3 experiments/generate_suite.py -o experiments/suite-LADDER/ -g LADDER --min-vars 3 --max-vars 10 --rep 5
