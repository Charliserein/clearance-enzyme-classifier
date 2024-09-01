
if [ $# -eq 0 ]; then
  echo "缺少输入文件"
  exit 1
fi

fasta_file=$1
cp ${fasta_file} work.fa

#### CNN
if [ ! -f work.fa ]; then
    echo "work.fa 文件不存在"
    exit 1
fi

python seq2pad.py work.fa fa.pt
if [ ! -f fa.pt ]; then
    echo "fa.pt 文件生成失败"
    exit 1
fi

python Rnn_test.py
rm -f fa.pt
echo "RNN finish(1/2)"

### NN
python iFeature-master/iFeature.py --file work.fa --type DPC --out 01
if [ -f 01 ]; then
    python scale.py

fi

if [ -f DPC.out ]; then
    python Bilstm_test.py > Bilstm_test.out
    rm -f 01 DPC.out
    echo "Bilstm finish(1/2)"

fi

## XGBOOST
python iFeature-master/iFeature.py --file work.fa --type CKSAAGP --out CKSAAGP.out
if [ -f CKSAAGP.out ]; then
    python xgb.py
#    rm -f CKSAAGP.out
    echo "XGBOOST (1/2)"
else
    echo "File CKSAAGP.out not found, XGBOOST step skipped."
    exit 1
fi

# Hard voting
paste xgb.out lstm.out lstm_test_class.out | awk '{print $1" "$2+$3+$4}' | awk '$2>1' | awk '{print $1}' > work.id
seqkit grep -f work.id work.fa > work-work.fa
### rm -f work.id

########################################################### Module 2
#### RNN
python seq2pad.py work-work.fa work.fa.pt
python Rnn.py
rm -f work.fa.pt
echo "RNN finish(2/2)"

#### XGBOOST
python iFeature-master/iFeature.py --file work-work.fa --type CKSAAGP --out CKSAAGP.out
if [ -f CKSAAGP.out ]; then
    python classifyxgb.py
#    rm -f CKSAAGP.out
    echo "XGBOOST (2/2)"
else
    echo "File CKSAAGP.out not found, XGBOOST step skipped."
    exit 1
fi

#### Bilstm
python iFeature-master/iFeature.py --file work-work.fa --type DPC --out 01
if [ -f 01 ]; then
    python scale.py

fi

if [ -f DPC.out ]; then
    python Bilstm.py
    echo "Bilstm-finish(2/2)"

    exit 1
fi

#### Soft voting
rm -f work-work.fa
python soft_vote.py
echo "soft voting finish"

# Clean up temporary files

# Output results
if [ -f final_result.csv ]; then
    cat final_result.csv
fi
