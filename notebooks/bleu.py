import subprocess as sp

def main():
  from argparse import ArgumentParser
  p = ArgumentParser()
  p.add_argument('hypref')
  args = p.parse_args()

  with open(args.hypref, 'r') as f:
    with open('.hyp.tmp', 'w') as h:
      with open('.ref.tmp', 'w') as r:
        for line in f:
          hyp, ref = line.split('\t')
          print(hyp.strip(), file=h)
          print(ref.strip(), file=r)
  result = sp.check_output(['sacrebleu', '-lc', '.ref.tmp', '--input', '.hyp.tmp'])
  print(result)

if __name__ == '__main__':
  main()
