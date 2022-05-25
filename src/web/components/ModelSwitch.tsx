import { css } from "@emotion/react";
import { Chip } from "@mui/material";

const models = ["Original MAE", "CelebA", "Places",]

export function ModelSwitch(props: {
    modelId: number;
    onChange: (newModelId: number) => void;
}) {
    return <div css={css({
        float: 'right',
        display: 'inline',
        marginTop: 14,
    })}>{
        models.map((model, i) => <Chip key={i}
            label={model}
            color="primary"
            variant={props.modelId === i ? "filled" : "outlined"}
            onClick={() => props.onChange(i)}
            css={css({
                marginRight: 8,
                height: 24
            })}/>)
    }</div>
}