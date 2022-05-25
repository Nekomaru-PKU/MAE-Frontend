import { useState, useEffect } from "react";
import { css } from '@emotion/react';
import { LinearProgress } from "@mui/material";

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
                    <td id='0, 0'><div css={css({
                        width: 300,
                        height: 300,
                        padding: 16,
                        background: `url(/upload/${props.uploadId})`,
                        backgroundSize: 'cover',
                    })}><ResultSubplotLabel text="Original"/></div></td>
                    <td id='0, 1'><div css={css({
                        width: 300,
                        height: 300,
                        padding: 16,
                        background: `url(/result/${props.uploadId}/masked.png)`,
                        backgroundSize: 'cover',
                    })}><ResultSubplotLabel text="Masked"/></div></td>
                </tr>
                <tr id='1'>
                    <td id='1, 0'><div css={css({
                        width: 300,
                        height: 300,
                        padding: 16,
                        background: `url(/result/${props.uploadId}/recon.png)`,
                        backgroundSize: 'cover',
                    })}><ResultSubplotLabel text="Reconstructed"/></div></td>
                    <td id='1, 1'><div css={css({
                        width: 300,
                        height: 300,
                        padding: 16,
                        background: `url(/result/${props.uploadId}/recon_visible.png)`,
                        backgroundSize: 'cover',
                    })}><ResultSubplotLabel text="Reconstructed + Visible"/></div></td>
                </tr>
            </tbody>
        </table>
    </div> : <>
        <LinearProgress css={css({
            height: 10,
        })}/>
    </>
}

function ResultSubplotLabel(props: {
    text: string,
}) {
    return <div css={css({
        width: 'fit-content',
        background: 'white',
        paddingLeft: 8,
        paddingRight: 8,
    })}>{props.text}</div>
}
