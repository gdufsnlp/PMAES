# **PMAES-Cross prompt AES**

Code for Paper "PMAES: Prompt-mapping Contrastive Learning for Cross-prompt Automated Essay Scoring"

ACL2023, Pages: 1489–1503

Yuan Chen, Xia Li


## **Model**
![](model.png)

## **Requirements**
- python==3.8
- torch==1.7.0+cu110
- tensorflow==2.5.0

## **Datasets**
ASAP and ASAP++ dataset in ./data (which are copy from [CTS](https://github.com/robert1ridley/cross-prompt-trait-scoring)).
## **Usage**

## **Training**
```
bash train.sh
```

## **Credits**
The code in this repository is based on [CTS](https://github.com/robert1ridley/cross-prompt-trait-scoring).

## **Citation**
```bibtex
@inproceedings{Chen2023PMAES,
    title = {PMAES: Prompt-mapping Contrastive Learning for Cross-prompt Automated Essay Scoring},
    author = {Yuan, Chen and Xia, Li},
    year = {2023},
    booktitle={Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
}
```
