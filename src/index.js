import { argv } from 'node:process';

function main(args) {
  const name = args[0] ?? 'world';
  console.log(`Hello, ${name}!`);
}

main(argv.slice(2));
