# train MIME

```
./run_sh/training.sh ${model_selection} ${ngpu} ${pretrained_loss_weight} ${weight_strict}
```

## Bedrooms
```
./run_sh/training.sh bedrooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 None True
```
## Livingrooms

```
./run_sh/training.sh living_rooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 None False
```

## Diningrooms

```
./run_sh/training.sh libraries_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 None False
```

## Libraries

```
./run_sh/training.sh diningrooms_freespaceFuse_AllContactHumans_RandomFreeNonOccupiedContactPEAnchorOnlyOne 1 None False
```
