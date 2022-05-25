const path = require('path');

module.exports = {
    mode: 'development',
    entry: path.resolve(__dirname, 'src/web/index.tsx'),
    devtool: 'inline-source-map',
    output: {
        path: path.resolve(__dirname, 'public'),
        filename: 'index.bundle.js'
    },
    module: {
        rules: [
            {
                include: path.resolve(__dirname, 'src/web/'),
                test: /\.tsx?$/,
                use: [
                    { loader: 'ts-loader', options: { transpileOnly: true } }
                ]
            },
        ],
    },
    resolve: {
        extensions: ['.js', '.ts', '.tsx'],
    },
    externals: {
        'react': 'React',
        'react-dom' : 'ReactDOM',
    },
    devServer: {
        static: path.resolve(__dirname, 'public'),
    }
};
