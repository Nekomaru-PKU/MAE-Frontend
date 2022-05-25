import { useState } from 'react';
import { css } from '@emotion/react'
import { Button, Step, StepContent, StepLabel, Stepper, useTheme } from "@mui/material";
import { ImageUpload } from './components/ImageUpload';
import { MaskEditor } from './components/MaskEditor';
import { ResultViewer } from './components/ResultViewer';

export function AppContent() {
    const theme = useTheme();
    const [step, setStep] = useState(0);
    const [uploadId, setUploadId] = useState('');
    const [maskBase64, setMaskBase64] = useState("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==");
    return <div css={css({
        width: "640px"
    })}>
        <h1 css={css({
            color: theme.palette.primary.main,
        })}>MAE Inpainting</h1>
        <Stepper activeStep={step} orientation="vertical">
            <Step>
                <StepLabel>Upload Image</StepLabel>
                <StepContent>
                    <ImageUpload onImageUpload={async image => {
                        const formData = new FormData();
                        formData.append('image', image, image.name);
                        const res = await fetch('/api/upload', {
                            method: 'POST',
                            body: formData,
                        });
                        const json = await res.json();
                        setUploadId(json.id);
                        setStep(1);
                        console.log(`Upload ID: ${json.id}`);
                    }}/>
                </StepContent>
            </Step>
            <Step>
                <StepLabel>Paint Mask</StepLabel>
                <StepContent>{ uploadId ? <>
                    <MaskEditor
                        uploadId={uploadId}
                        maskBase64={maskBase64}
                        onChange={newMaskBase64 => setMaskBase64(newMaskBase64)} />
                    <Button variant="contained" css={css({
                        float: 'right',
                        height: 28,
                        marginTop: 8,
                        borderRadius: 14,
                    })} onClick={() => {
                        fetch("/api/run", {
                            method: "POST",
                            headers: {
                              'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                id: uploadId,
                                modelId: "0",
                                maskBase64: maskBase64,
                            })
                        });
                        setStep(2);
                    }}>Next</Button>
                </> : (<></>) }</StepContent>
            </Step>
            <Step>
                <StepLabel>Reconstruct</StepLabel>
                <StepContent>
                    <ResultViewer uploadId={uploadId} />
                </StepContent>
            </Step>
        </Stepper>
    </div>
}
