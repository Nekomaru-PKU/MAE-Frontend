import { useState, useEffect } from "react";
import { css } from '@emotion/react';
import { Grow, LinearProgress } from "@mui/material";

export function ResultViewer(props: {
    uploadId: string;
}) {
    const [ready, setReady] = useState(false);
    useEffect(() => {
        async function checkReady() {
            const res = await fetch(`/result/${props.uploadId}/recon.png`, {
                method: "HEAD",
            });
            if (res.ok) {
                setReady(true);
                clearInterval(handle);
            }
        }
        const handle = setInterval(checkReady, 1000);
        return () => clearInterval(handle);
    })
    return ready ? <div>
        <table>
            <tbody>
                <tr id='0'>
                    <td id='0, 0'><ResultSubplot ready={ready} sequence={0}
                        label="Original"
                        imageUrl={`/upload/${props.uploadId}`}/></td>
                    <td id='0, 1'><ResultSubplot ready={ready} sequence={1}
                        label="Masked"
                        imageUrl={`/result/${props.uploadId}/masked.png`}/></td>
                </tr>
                <tr id='1'>
                    <td id='1, 0'><ResultSubplot ready={ready} sequence={2}
                        label="Reconstructed"
                        imageUrl={`/result/${props.uploadId}/recon.png`}/></td>
                    <td id='1, 1'><ResultSubplot ready={ready} sequence={3}
                        label="Reconstructed + Visible"
                        imageUrl={`/result/${props.uploadId}/recon_visible.png`}/></td>
                </tr>
            </tbody>
        </table>
    </div> : <>
        <LinearProgress css={css({
            height: 10,
        })}/>
    </>
}

function ResultSubplot(props: {
    ready: boolean;
    label: string;
    imageUrl: string;
    sequence: number;
}) {
    return <Grow in={props.ready} timeout={1000 * props.sequence}>
        <div css={css({
            width: 300,
            height: 300,
            padding: 16,
            background: `url(${props.imageUrl})`,
            backgroundSize: 'cover',
        })}>
            <div css={css({
                width: 'fit-content',
                background: 'white',
                paddingLeft: 8,
                paddingRight: 8,
            })}>{props.label}</div>
        </div>
    </Grow> 
}
